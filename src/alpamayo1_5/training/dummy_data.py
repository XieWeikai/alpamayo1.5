"""Dummy dataset and collator for Alpamayo 1.5 training smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from alpamayo1_5.models.base_model import SPECIAL_TOKENS, TRAJ_TOKEN


def yaw_to_rotation_matrices(yaw: torch.Tensor) -> torch.Tensor:
    """Convert yaw angles to batched 3x3 rotation matrices."""

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    zeros = torch.zeros_like(cos_yaw)
    ones = torch.ones_like(cos_yaw)
    return torch.stack(
        [
            torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
            torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )


def _compute_yaw_from_xy(xy: torch.Tensor) -> torch.Tensor:
    deltas = xy[1:] - xy[:-1]
    deltas = torch.cat([deltas[:1], deltas], dim=0)
    return torch.atan2(deltas[:, 1], deltas[:, 0].clamp_min(1e-4))


@dataclass(slots=True)
class DummyTrajectorySample:
    prompt_text: str
    cot_text: str
    ego_history_xyz: torch.Tensor
    ego_history_rot: torch.Tensor
    ego_future_xyz: torch.Tensor
    ego_future_rot: torch.Tensor


class DummyAlpamayoDataset(Dataset[DummyTrajectorySample]):
    """Deterministic dummy dataset for validating the training pipeline."""

    def __init__(
        self,
        *,
        num_samples: int = 64,
        history_steps: int = 16,
        future_steps: int = 64,
    ) -> None:
        self.num_samples = num_samples
        self.history_steps = history_steps
        self.future_steps = future_steps

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> DummyTrajectorySample:
        history_xyz, history_rot, future_xyz, future_rot = self._build_trajectory(index)
        prompt_text = (
            f"Scene {index}: multi-camera driving context with a gentle curvature and a speed "
            f"change. Use the trajectory history tokens as extra context, then predict the next "
            f"reasoning trace and future plan."
        )
        cot_text = (
            f"Focus on lane keeping, smooth speed control, and curvature pattern {index % 5}. "
            f"Keep the plan consistent with the observed history."
        )
        return DummyTrajectorySample(
            prompt_text=prompt_text,
            cot_text=cot_text,
            ego_history_xyz=history_xyz,
            ego_history_rot=history_rot,
            ego_future_xyz=future_xyz,
            ego_future_rot=future_rot,
        )

    def _build_trajectory(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        history_t = torch.linspace(-(self.history_steps - 1), 0, self.history_steps, dtype=torch.float32)
        future_t = torch.linspace(1, self.future_steps, self.future_steps, dtype=torch.float32)

        base_speed = 0.12 + 0.01 * (index % 7)
        curvature = 0.0025 * ((index % 9) - 4)
        lateral_bias = 0.03 * ((index % 3) - 1)

        history_x = base_speed * history_t
        history_y = curvature * history_t.square() + lateral_bias
        future_x = base_speed * future_t
        future_y = curvature * future_t.square() + lateral_bias

        history_xyz = torch.stack([history_x, history_y, torch.zeros_like(history_x)], dim=-1)
        future_xyz = torch.stack([future_x, future_y, torch.zeros_like(future_x)], dim=-1)

        history_yaw = _compute_yaw_from_xy(history_xyz[:, :2])
        future_yaw = _compute_yaw_from_xy(torch.cat([history_xyz[-1:, :2], future_xyz[:, :2]], dim=0))[1:]

        history_rot = yaw_to_rotation_matrices(history_yaw)
        future_rot = yaw_to_rotation_matrices(future_yaw)

        return history_xyz.unsqueeze(0), history_rot.unsqueeze(0), future_xyz.unsqueeze(0), future_rot.unsqueeze(0)


class DummyAlpamayoCollator:
    """Build token sequences, labels, and trajectory tensors for dummy training."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        traj_tokenizer: Any,
        future_token_start_idx: int,
        tokens_per_history_traj: int,
        expected_future_tokens: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.traj_tokenizer = traj_tokenizer
        self.future_token_start_idx = future_token_start_idx
        self.tokens_per_history_traj = tokens_per_history_traj
        self.expected_future_tokens = expected_future_tokens
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.history_token_id = tokenizer.convert_tokens_to_ids(TRAJ_TOKEN["history"])
        self.cot_start_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["cot_start"])
        self.cot_end_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["cot_end"])
        self.future_start_id = tokenizer.convert_tokens_to_ids(TRAJ_TOKEN["future_start"])
        self.future_end_id = tokenizer.convert_tokens_to_ids(TRAJ_TOKEN["future_end"])

    def __call__(self, items: list[DummyTrajectorySample]) -> dict[str, torch.Tensor]:
        input_id_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        output_start_positions: list[int] = []
        sequence_lengths: list[int] = []

        history_xyz = []
        history_rot = []
        future_xyz = []
        future_rot = []

        for item in items:
            prompt_ids = self._encode_prompt(item.prompt_text)
            output_start_positions.append(len(prompt_ids))

            future_ids = self.traj_tokenizer.encode(
                hist_xyz=item.ego_history_xyz,
                hist_rot=item.ego_history_rot,
                fut_xyz=item.ego_future_xyz,
                fut_rot=item.ego_future_rot,
            ).squeeze(0)
            if future_ids.numel() != self.expected_future_tokens:
                raise ValueError(
                    f"Expected {self.expected_future_tokens} future tokens, got {future_ids.numel()}"
                )
            future_ids = future_ids + self.future_token_start_idx

            response_ids = self._encode_response(item.cot_text, future_ids)
            full_ids = torch.tensor(prompt_ids + response_ids, dtype=torch.long)
            labels = full_ids.clone()
            labels[: output_start_positions[-1]] = -100

            input_id_rows.append(full_ids)
            label_rows.append(labels)
            sequence_lengths.append(full_ids.numel())

            history_xyz.append(item.ego_history_xyz)
            history_rot.append(item.ego_history_rot)
            future_xyz.append(item.ego_future_xyz)
            future_rot.append(item.ego_future_rot)

        max_length = max(sequence_lengths)
        input_ids = torch.full((len(items), max_length), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(items), max_length), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(items), max_length), dtype=torch.long)
        for row_idx, (row_ids, row_labels) in enumerate(zip(input_id_rows, label_rows, strict=True)):
            seq_len = row_ids.numel()
            input_ids[row_idx, :seq_len] = row_ids
            labels[row_idx, :seq_len] = row_labels
            attention_mask[row_idx, :seq_len] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "output_start_positions": torch.tensor(output_start_positions, dtype=torch.long),
            "sequence_lengths": torch.tensor(sequence_lengths, dtype=torch.long),
            "ego_history_xyz": torch.stack(history_xyz, dim=0),
            "ego_history_rot": torch.stack(history_rot, dim=0),
            "ego_future_xyz": torch.stack(future_xyz, dim=0),
            "ego_future_rot": torch.stack(future_rot, dim=0),
        }

    def _encode_prompt(self, prompt_text: str) -> list[int]:
        prompt_ids: list[int] = []
        if self.bos_token_id is not None:
            prompt_ids.append(self.bos_token_id)
        prompt_ids.extend(
            self.tokenizer.encode(
                f"User:\n{prompt_text}\nTrajectory history:\n",
                add_special_tokens=False,
            )
        )
        prompt_ids.extend([self.history_token_id] * self.tokens_per_history_traj)
        prompt_ids.extend(
            self.tokenizer.encode(
                "\nAssistant: reason carefully and then emit trajectory tokens.\n",
                add_special_tokens=False,
            )
        )
        return prompt_ids

    def _encode_response(self, cot_text: str, future_ids: torch.Tensor) -> list[int]:
        response_ids = [self.cot_start_id]
        response_ids.extend(self.tokenizer.encode(cot_text, add_special_tokens=False))
        response_ids.extend([self.cot_end_id, self.future_start_id])
        response_ids.extend(future_ids.tolist())
        response_ids.append(self.future_end_id)
        if self.eos_token_id is not None:
            response_ids.append(self.eos_token_id)
        return response_ids
