from __future__ import annotations

import torch

from alpamayo1_5.models.base_model import SPECIAL_TOKENS, TRAJ_TOKEN
from alpamayo1_5.training.dummy_data import DummyAlpamayoCollator, DummyTrajectorySample
from alpamayo1_5.training.objectives import (
    build_flow_matching_inputs,
    resolve_expert_offsets,
    sample_low_timestep_beta,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self._token_to_id = {
            TRAJ_TOKEN['history']: 10,
            SPECIAL_TOKENS['cot_start']: 11,
            SPECIAL_TOKENS['cot_end']: 12,
            TRAJ_TOKEN['future_start']: 13,
            TRAJ_TOKEN['future_end']: 14,
        }
        self._next_id = 100

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_to_id[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids = []
        for token in text.split():
            if token not in self._token_to_id:
                self._token_to_id[token] = self._next_id
                self._next_id += 1
            ids.append(self._token_to_id[token])
        return ids


class FakeTrajTokenizer:
    def encode(self, **_: object) -> torch.Tensor:
        return torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)


def make_sample(index: int) -> DummyTrajectorySample:
    hist_xyz = torch.zeros(1, 2, 3)
    hist_rot = torch.eye(3).repeat(1, 2, 1, 1)
    fut_xyz = torch.ones(1, 2, 3) * index
    fut_rot = torch.eye(3).repeat(1, 2, 1, 1)
    return DummyTrajectorySample(
        prompt_text=f'prompt {index}',
        cot_text=f'cot {index}',
        ego_history_xyz=hist_xyz,
        ego_history_rot=hist_rot,
        ego_future_xyz=fut_xyz,
        ego_future_rot=fut_rot,
    )


def test_low_timestep_beta_sampler_biases_toward_small_values() -> None:
    timesteps = sample_low_timestep_beta(
        batch_size=4096,
        ndim=3,
        device=torch.device('cpu'),
        dtype=torch.float32,
        upper_bound=0.999,
        alpha=1.5,
        beta=1.0,
    )
    assert timesteps.min().item() >= 0.0
    assert timesteps.max().item() <= 0.999
    assert timesteps.mean().item() < 0.45


def test_build_flow_matching_inputs_matches_linear_path() -> None:
    target = torch.ones(2, 4, 3)
    timesteps = torch.full((2, 1, 1), 0.25)
    noise = torch.zeros_like(target)
    noisy_action, vector_field = build_flow_matching_inputs(target, timesteps, noise=noise)
    assert torch.allclose(noisy_action, torch.full_like(target, 0.25))
    assert torch.allclose(vector_field, torch.ones_like(target))


def test_resolve_expert_offsets_switches_between_modes() -> None:
    sequence_lengths = torch.tensor([10, 12])
    output_starts = torch.tensor([4, 5])
    assert torch.equal(resolve_expert_offsets('full_kv', sequence_lengths, output_starts), sequence_lengths)
    assert torch.equal(resolve_expert_offsets('input_only', sequence_lengths, output_starts), output_starts)


def test_collator_masks_prompt_tokens_and_keeps_output_tokens() -> None:
    collator = DummyAlpamayoCollator(
        tokenizer=FakeTokenizer(),
        traj_tokenizer=FakeTrajTokenizer(),
        future_token_start_idx=200,
        tokens_per_history_traj=4,
        expected_future_tokens=6,
    )
    batch = collator([make_sample(1), make_sample(2)])
    first_output_start = batch['output_start_positions'][0].item()
    assert batch['input_ids'].shape[0] == 2
    assert torch.all(batch['labels'][0, :first_output_start] == -100)
    assert torch.any(batch['labels'][0, first_output_start:] != -100)
    assert batch['sequence_lengths'][0].item() <= batch['input_ids'].shape[1]
