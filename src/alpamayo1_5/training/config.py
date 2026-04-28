"""Configuration for the standalone dummy training entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ExpertContextMode = Literal["full_kv", "input_only"]


@dataclass(slots=True)
class DummyTrainingConfig:
    """Configuration for the standalone dummy trainer."""

    model_path: str = "/data-25T/models/Alpamayo-1.5-10B"
    vlm_processor_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    output_dir: Path = Path("outputs/dummy_training")
    max_steps: int = 10
    batch_size: int = 1
    grad_accum_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_workers: int = 0
    seed: int = 42
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    expert_context_mode: ExpertContextMode = "input_only"
    vlm_loss_weight: float = 1.0
    expert_loss_weight: float = 1.0
    flow_timestep_ceiling: float = 0.999
    flow_beta_alpha: float = 1.5
    flow_beta_beta: float = 1.0
    dummy_num_samples: int = 64
    dummy_history_steps: int = 16
    dummy_future_steps: int = 64
    log_every: int = 1
    enforce_fsdp_4gpu: bool = True
    save_final_state: bool = True
