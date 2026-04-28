# Milestone 02 - 测试设计与结果

## 独立测试 Agent

- Agent ID: `019db993-9414-77a0-aacc-ab8889e615a2`
- Nickname: `Boole`

## 实际执行内容

1. `PYTHONPATH=src /data/envs/alpamayo/bin/python -m compileall src/alpamayo1_5/training scripts/train_dummy_fsdp.py tests/test_dummy_training_utils.py`
2. 手工执行以下测试函数：
   - `test_low_timestep_beta_sampler_biases_toward_small_values`
   - `test_build_flow_matching_inputs_matches_linear_path`
   - `test_resolve_expert_offsets_switches_between_modes`
   - `test_collator_masks_prompt_tokens_and_keeps_output_tokens`
3. 静态检查 `configs/accelerate/fsdp_4gpu_dummy.yaml`
4. `PYTHONPATH=src /data/envs/alpamayo/bin/python scripts/train_dummy_fsdp.py --help`

## 通过项

- `compileall` 通过
- 4 个轻量单测通过
- FSDP YAML 静态检查通过
- CLI 帮助输出通过

## 本地主 agent 同步验证

- 手工单测本地再次执行通过
- 运行过程中出现 `torch.cuda` 初始化 warning，但不影响 CPU/静态级验证

## 仍未覆盖的高风险项

- 未真实加载 `/data-25T/models/Alpamayo-1.5-10B`
- 未真实执行 `DummyTrainingModule.forward`
- 未真实执行 `accelerate launch --config_file configs/accelerate/fsdp_4gpu_dummy.yaml ...`
- 未真实验证 `past_key_values` / `rope_deltas` / 4D attention mask 在训练 forward 路径下的兼容性
- 未真实验证 `SIZE_BASED_WRAP` 在 10B 模型上的显存行为
