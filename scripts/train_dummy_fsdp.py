"""Train Alpamayo 1.5 on a dummy dataset with Accelerate + FSDP."""

from __future__ import annotations

import argparse
from pathlib import Path

from alpamayo1_5.training.config import DummyTrainingConfig
from alpamayo1_5.training.runner import run_dummy_training


def parse_args() -> DummyTrainingConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/data-25T/models/Alpamayo-1.5-10B")
    parser.add_argument("--vlm-processor-name-or-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dummy_training"))
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--expert-context-mode", choices=["full_kv", "input_only"], default="input_only")
    parser.add_argument("--vlm-loss-weight", type=float, default=1.0)
    parser.add_argument("--expert-loss-weight", type=float, default=1.0)
    parser.add_argument("--flow-timestep-ceiling", type=float, default=0.999)
    parser.add_argument("--flow-beta-alpha", type=float, default=1.5)
    parser.add_argument("--flow-beta-beta", type=float, default=1.0)
    parser.add_argument("--dummy-num-samples", type=int, default=64)
    parser.add_argument("--dummy-history-steps", type=int, default=16)
    parser.add_argument("--dummy-future-steps", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--no-enforce-fsdp-4gpu", action="store_true")
    parser.add_argument("--no-save-final-state", action="store_true")
    args = parser.parse_args()
    return DummyTrainingConfig(
        model_path=args.model_path,
        vlm_processor_name_or_path=args.vlm_processor_name_or_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        expert_context_mode=args.expert_context_mode,
        vlm_loss_weight=args.vlm_loss_weight,
        expert_loss_weight=args.expert_loss_weight,
        flow_timestep_ceiling=args.flow_timestep_ceiling,
        flow_beta_alpha=args.flow_beta_alpha,
        flow_beta_beta=args.flow_beta_beta,
        dummy_num_samples=args.dummy_num_samples,
        dummy_history_steps=args.dummy_history_steps,
        dummy_future_steps=args.dummy_future_steps,
        log_every=args.log_every,
        enforce_fsdp_4gpu=not args.no_enforce_fsdp_4gpu,
        save_final_state=not args.no_save_final_state,
    )


def main() -> None:
    run_dummy_training(parse_args())


if __name__ == "__main__":
    main()
