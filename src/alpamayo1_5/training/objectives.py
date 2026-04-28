"""Objective helpers for dummy Alpamayo training."""

from __future__ import annotations

import torch


def sample_low_timestep_beta(
    batch_size: int,
    ndim: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    upper_bound: float = 0.999,
    alpha: float = 1.5,
    beta: float = 1.0,
) -> torch.Tensor:
    """Sample low timesteps with the Beta-based schedule described in the task."""

    base_dist = torch.distributions.Beta(alpha, beta)
    beta_sample = base_dist.sample((batch_size,)).to(device=device, dtype=dtype)
    tau = upper_bound * (1.0 - beta_sample)
    return tau.view(batch_size, *([1] * (ndim - 1)))


def build_flow_matching_inputs(
    target_action: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build noisy actions and vector-field targets for flow matching."""

    if noise is None:
        noise = torch.randn_like(target_action)
    x_t = (1.0 - timesteps) * noise + timesteps * target_action
    vector_field = target_action - noise
    return x_t, vector_field


def resolve_expert_offsets(
    mode: str,
    sequence_lengths: torch.Tensor,
    output_start_positions: torch.Tensor,
) -> torch.Tensor:
    """Resolve which prefix of the VLM KV cache the expert can attend to."""

    if mode == "full_kv":
        return sequence_lengths
    if mode == "input_only":
        return output_start_positions
    raise ValueError(f"Unsupported expert context mode: {mode}")
