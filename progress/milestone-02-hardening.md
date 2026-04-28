# Milestone 02 - 启动约束、测试与交付硬化

## 目标

根据第一轮独立 review/test 的反馈，补齐 4 卡 FSDP 约束、token 合约检查、测试和说明文档，使这套 dummy training 代码更接近可直接运行的交付状态。

## 本里程碑新增/完善内容

- 新增 `configs/accelerate/fsdp_4gpu_dummy.yaml`
  - 固定 `distributed_type: FSDP`
  - 固定 `num_processes: 4`
  - 固定 `mixed_precision: bf16`
- 在 `runner.py` 中新增启动约束：
  - 默认强制 `FSDP + 4 processes + 至少 4 张可见 GPU`
  - 提供 `--no-enforce-fsdp-4gpu` 便于调试
- 在 `runner.py` 中新增模型合约检查：
  - 检查关键 special token 是否存在
  - 检查 history token 数与 `tokens_per_history_traj` 一致
  - 检查 future token 数与 `tokens_per_future_traj` 一致
- 新增轻量测试文件 `tests/test_dummy_training_utils.py`
- 新增使用说明 `TRAINING_DUMMY.md`
- 新增 milestone 文档记录到 `progress/` 与 `test/`

## 本地验证

已完成的本地轻量验证：

- `compileall` 通过
- 手工执行 4 个轻量单测通过
- `scripts/train_dummy_fsdp.py --help` 输出正常
- FSDP YAML 配置静态检查通过

## 第二轮独立 Review

- Review agent：`019db993-5624-7632-bd51-c7c415c5ef2d`（Archimedes）
- 结论摘要：
  - 从代码设计角度，本版本已满足“4 卡 + FSDP + dummy 训练”需求。
  - 本轮未发现新的阻塞问题。
  - 最大剩余风险是仍未完成真实 `10B + 4卡` smoke。

## 第二轮独立测试设计/执行

- Test agent：`019db993-9414-77a0-aacc-ab8889e615a2`（Boole）
- 执行摘要：
  - `compileall`
  - 手工调用 4 个测试函数
  - 静态检查 FSDP YAML
  - CLI `--help`
- 结果摘要：
  - 以上检查全部通过
  - 未做真实 10B 模型 forward 与真实 4 卡 FSDP 启动

## 当前状态

- 作为代码交付：已完成
- 作为“已在当前机器上实机证明可训练”的版本：仍差最后一步真实 4 卡 smoke
