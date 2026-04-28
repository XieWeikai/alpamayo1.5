"""Standalone runner for dummy Alpamayo training."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType, set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader

from alpamayo1_5.config import Alpamayo1_5Config
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.training.config import DummyTrainingConfig
from alpamayo1_5.training.dummy_data import DummyAlpamayoCollator, DummyAlpamayoDataset
from alpamayo1_5.training.module import DummyTrainingModule


def _to_torch_dtype(dtype_name: str) -> torch.dtype:
    try:
        return getattr(torch, dtype_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def _enable_fsdp_ram_efficient_loading() -> None:
    try:
        from accelerate.utils import enable_fsdp_ram_efficient_loading

        enable_fsdp_ram_efficient_loading()
    except Exception:
        pass


def _load_training_model(config: DummyTrainingConfig) -> Alpamayo1_5:
    config_path = Path(config.model_path) / "config.json"
    with config_path.open() as f:
        config_dict = json.load(f)
    config_dict["vlm_name_or_path"] = config.vlm_processor_name_or_path
    config_dict["attn_implementation"] = config.attn_implementation
    hf_config = Alpamayo1_5Config(**config_dict)
    return Alpamayo1_5.from_pretrained(
        config.model_path,
        config=hf_config,
        torch_dtype=_to_torch_dtype(config.dtype),
    )


def _assert_launch_contract(config: DummyTrainingConfig, accelerator: Accelerator) -> None:
    if not config.enforce_fsdp_4gpu:
        return
    if accelerator.distributed_type != DistributedType.FSDP:
        raise RuntimeError(
            "This entrypoint is meant to run with Accelerate FSDP. Use the provided 4-GPU "
            "config or pass --no-enforce-fsdp-4gpu for debugging."
        )
    if accelerator.num_processes != 4:
        raise RuntimeError(
            f"Expected 4 processes for the requested setup, got {accelerator.num_processes}."
        )
    if torch.cuda.device_count() < 4:
        raise RuntimeError(
            f"Expected at least 4 visible CUDA devices, got {torch.cuda.device_count()}."
        )


def _validate_model_contracts(model: Alpamayo1_5, dataset: DummyAlpamayoDataset) -> None:
    sample = dataset[0]
    tokenizer = model.tokenizer
    unk_token_id = getattr(tokenizer, "unk_token_id", None)

    required_token_map = {
        "traj_history": "<|traj_history|>",
        "traj_future_start": "<|traj_future_start|>",
        "traj_future_end": "<|traj_future_end|>",
        "cot_start": "<|cot_start|>",
        "cot_end": "<|cot_end|>",
    }
    for name, token in required_token_map.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or (unk_token_id is not None and token_id == unk_token_id):
            raise RuntimeError(f"Missing required tokenizer token: {name}")

    history_tokens = model.hist_traj_tokenizer.encode(
        hist_xyz=sample.ego_history_xyz,
        hist_rot=sample.ego_history_rot,
        fut_xyz=sample.ego_history_xyz,
        fut_rot=sample.ego_history_rot,
    )
    if history_tokens.shape[-1] != model.config.tokens_per_history_traj:
        raise RuntimeError(
            "History token count mismatch: "
            f"expected {model.config.tokens_per_history_traj}, got {history_tokens.shape[-1]}"
        )

    future_tokens = model.traj_tokenizer.encode(
        hist_xyz=sample.ego_history_xyz,
        hist_rot=sample.ego_history_rot,
        fut_xyz=sample.ego_future_xyz,
        fut_rot=sample.ego_future_rot,
    )
    if future_tokens.shape[-1] != model.config.tokens_per_future_traj:
        raise RuntimeError(
            "Future token count mismatch: "
            f"expected {model.config.tokens_per_future_traj}, got {future_tokens.shape[-1]}"
        )

def run_dummy_training(config: DummyTrainingConfig) -> None:
    """Run the standalone dummy training loop."""

    accelerator = Accelerator(
        gradient_accumulation_steps=config.grad_accum_steps,
        mixed_precision="bf16" if config.dtype == "bfloat16" else "no",
    )
    if accelerator.distributed_type == DistributedType.FSDP:
        _enable_fsdp_ram_efficient_loading()

    _assert_launch_contract(config, accelerator)
    set_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    output_dir = Path(config.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_training_model(config)
    model.train()

    dataset = DummyAlpamayoDataset(
        num_samples=config.dummy_num_samples,
        history_steps=config.dummy_history_steps,
        future_steps=config.dummy_future_steps,
    )
    _validate_model_contracts(model, dataset)
    collator = DummyAlpamayoCollator(
        tokenizer=model.tokenizer,
        traj_tokenizer=model.traj_tokenizer,
        future_token_start_idx=model.future_token_start_idx,
        tokens_per_history_traj=model.config.tokens_per_history_traj,
        expected_future_tokens=model.config.tokens_per_future_traj,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    training_module = DummyTrainingModule(model=model, config=config)
    optimizer = AdamW(training_module.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    training_module, optimizer, dataloader = accelerator.prepare(training_module, optimizer, dataloader)

    step = 0
    while step < config.max_steps:
        for batch in dataloader:
            with accelerator.accumulate(training_module):
                with accelerator.autocast():
                    losses = training_module(batch)
                accelerator.backward(losses["loss"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process and step % config.log_every == 0:
                accelerator.print(
                    f"step={step} "
                    f"loss={losses['loss'].detach().float().item():.4f} "
                    f"vlm_loss={losses['vlm_loss'].float().item():.4f} "
                    f"expert_loss={losses['expert_loss'].float().item():.4f}"
                )

            step += 1
            if step >= config.max_steps:
                break

    accelerator.wait_for_everyone()
    if config.save_final_state:
        accelerator.save_state(str(output_dir / "checkpoint-final"))
