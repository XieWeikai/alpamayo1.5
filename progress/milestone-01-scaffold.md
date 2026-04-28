# Milestone 01 - 训练骨架落地

## 目标

在不修改现有推理逻辑的前提下，独立增加一套可用于 Alpamayo 1.5 dummy training 的最小训练代码。

## 本里程碑新增内容

- 新增 `src/alpamayo1_5/training/` 子模块，拆分为：
  - `config.py`：训练配置
  - `dummy_data.py`：dummy dataset + collator
  - `objectives.py`：flow matching timestep 采样与辅助函数
  - `module.py`：联合 VLM NTP loss 与 action expert flow matching loss
  - `runner.py`：accelerate 训练循环
- 新增 `scripts/train_dummy_fsdp.py` 作为入口脚本。
- 训练设计满足两类目标：
  - VLM 输出部分做 next-token prediction
  - action expert 用 flow matching 学习轨迹 action
- action expert 上下文支持两种 KV cache 模式：
  - `input_only`
  - `full_kv`

## 关键设计点

- 仅新增文件，不改原有模型推理代码。
- dummy batch 中显式包含：
  - 输入 prompt
  - CoT 文本输出
  - trajectory token 输出
  - history / future trajectory 张量
- flow matching timestep 采样采用低 timestep 偏置的 Beta 形式，而非均匀采样。
- 使用训练包装模块 `DummyTrainingModule`，避免 FSDP 下直接绕过根模块调用子模块 forward。

## 独立 Review

- Review agent：`019db98e-92e8-7c02-a1a5-4bb38df2e0b5`（Mencius）
- Review 结论摘要：
  - 指出脚本尚未强约束 `4 GPU + FSDP` 启动条件。
  - 指出需要显式检查 history/future trajectory token 数与配置是否一致。
  - 建议增加特殊 token 存在性检查与更清晰的报错。

## 独立测试设计/执行

- Test agent：`019db98e-bdab-70c3-bf56-554e41c26d69`（Epicurus）
- 建议的测试范围：
  - `compileall`
  - flow matching 纯函数
  - dummy dataset/collator
  - CLI 参数解析
- 主要风险提醒：
  - 还未验证真实 10B 模型 forward
  - 还未验证真实 4 卡 FSDP smoke

## 下一步

在第二个里程碑中补齐启动约束、token 合约检查、4 卡 FSDP 启动配置、测试文件与使用文档。
