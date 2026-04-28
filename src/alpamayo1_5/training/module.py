"""Training wrapper module that combines VLM and action-expert objectives."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpamayo1_5.training.config import DummyTrainingConfig
from alpamayo1_5.training.objectives import (
    build_flow_matching_inputs,
    resolve_expert_offsets,
    sample_low_timestep_beta,
)


class DummyTrainingModule(nn.Module):
    """Wrap the pretrained model with a minimal dual-objective forward pass."""

    def __init__(self, model: Alpamayo1_5, config: DummyTrainingConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        fused_input_ids = self.model.fuse_traj_tokens(
            batch["input_ids"],
            {
                "ego_history_xyz": batch["ego_history_xyz"],
                "ego_history_rot": batch["ego_history_rot"],
            },
        )

        vlm_outputs = self.model.vlm(
            input_ids=fused_input_ids,
            attention_mask=batch["attention_mask"],
            use_cache=True,
            return_dict=True,
        )
        logits = vlm_outputs.logits[:, :-1].contiguous()
        labels = batch["labels"][:, 1:].contiguous()
        vlm_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )

        expert_loss = self._compute_expert_loss(vlm_outputs, batch)
        total_loss = (
            self.config.vlm_loss_weight * vlm_loss
            + self.config.expert_loss_weight * expert_loss
        )
        return {
            "loss": total_loss,
            "vlm_loss": vlm_loss.detach(),
            "expert_loss": expert_loss.detach(),
        }

    def _compute_expert_loss(
        self,
        vlm_outputs: Any,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        with torch.autocast(device_type=batch["ego_history_xyz"].device.type, enabled=False):
            target_action = self.model.action_space.traj_to_action(
                traj_history_xyz=batch["ego_history_xyz"][:, 0].float(),
                traj_history_rot=batch["ego_history_rot"][:, 0].float(),
                traj_future_xyz=batch["ego_future_xyz"][:, 0].float(),
                traj_future_rot=batch["ego_future_rot"][:, 0].float(),
            )
        target_action = target_action.to(dtype=self.model.action_out_proj.weight.dtype)
        timesteps = sample_low_timestep_beta(
            batch_size=target_action.shape[0],
            ndim=target_action.ndim,
            device=target_action.device,
            dtype=target_action.dtype,
            upper_bound=self.config.flow_timestep_ceiling,
            alpha=self.config.flow_beta_alpha,
            beta=self.config.flow_beta_beta,
        )
        noisy_action, target_vector_field = build_flow_matching_inputs(target_action, timesteps)

        prompt_cache = vlm_outputs.past_key_values
        kv_cache_seq_len = prompt_cache.get_seq_length()
        rope_deltas = getattr(self.model.vlm.model, "rope_deltas", None)
        if rope_deltas is None:
            rope_deltas = torch.zeros(
                target_action.shape[0],
                1,
                device=target_action.device,
                dtype=torch.long,
            )
        else:
            rope_deltas = rope_deltas.to(device=target_action.device)
            if rope_deltas.ndim == 1:
                rope_deltas = rope_deltas.unsqueeze(-1)

        offsets = resolve_expert_offsets(
            mode=self.config.expert_context_mode,
            sequence_lengths=batch["sequence_lengths"],
            output_start_positions=batch["output_start_positions"],
        )
        n_diffusion_tokens = self.model.action_space.get_action_space_dims()[0]
        position_ids, attention_mask = self.model._build_expert_pos_ids_and_attn_mask(
            offset=offsets,
            rope_deltas=rope_deltas,
            kv_cache_seq_len=kv_cache_seq_len,
            n_diffusion_tokens=n_diffusion_tokens,
            b_star=target_action.shape[0],
            device=target_action.device,
            prefix_mask=batch["attention_mask"],
        )

        future_token_embeds = self.model.action_in_proj(noisy_action, timesteps)
        if future_token_embeds.dim() == 2:
            future_token_embeds = future_token_embeds.view(target_action.shape[0], n_diffusion_tokens, -1)

        forward_kwargs = {}
        if self.model.config.expert_non_causal_attention:
            forward_kwargs["is_causal"] = False

        expert_outputs = self.model.expert(
            inputs_embeds=future_token_embeds,
            position_ids=position_ids,
            past_key_values=prompt_cache,
            attention_mask=attention_mask,
            use_cache=True,
            **forward_kwargs,
        )
        prompt_cache.crop(kv_cache_seq_len)
        pred_vector_field = self.model.action_out_proj(expert_outputs.last_hidden_state[:, -n_diffusion_tokens:])
        pred_vector_field = pred_vector_field.reshape_as(target_action)
        return F.mse_loss(pred_vector_field, target_vector_field)
