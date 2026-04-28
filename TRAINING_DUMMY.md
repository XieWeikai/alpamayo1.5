# Dummy FSDP Training for Alpamayo 1.5

这个仓库新增了一套完全独立的 dummy training 代码，目标是最小化侵入地验证 Alpamayo 1.5 可以走通训练。

## 代码位置

- `scripts/train_dummy_fsdp.py`：训练入口
- `src/alpamayo1_5/training/config.py`：训练配置
- `src/alpamayo1_5/training/dummy_data.py`：dummy dataset + collator
- `src/alpamayo1_5/training/objectives.py`：flow matching 采样与辅助函数
- `src/alpamayo1_5/training/module.py`：联合 VLM loss + action expert loss
- `src/alpamayo1_5/training/runner.py`：accelerate 训练循环
- `configs/accelerate/fsdp_4gpu_dummy.yaml`：4 卡 FSDP 启动配置

## 训练目标

1. **VLM 部分**
   - 输入包含 prompt 和 history trajectory token 占位符。
   - 输出部分包含 `CoT` 与未来 trajectory tokens。
   - 训练时仅对输出部分计算 next-token prediction loss。

2. **Action expert 部分**
   - 使用 flow matching 训练轨迹 action expert。
   - timestep 采样不是均匀分布，而是
     `tau = s * (1 - Beta(1.5, 1.0))`，其中 `s=0.999`。
   - `--expert-context-mode` 支持两种模式：
     - `input_only`：只能看 VLM 输入部分 KV cache
     - `full_kv`：可以看 VLM 全部 KV cache

## 运行方式

先激活用户指定环境：

```bash
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate /data/envs/alpamayo
```

然后用 accelerate 启动 4 卡 FSDP：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch \
  --config_file configs/accelerate/fsdp_4gpu_dummy.yaml \
  scripts/train_dummy_fsdp.py \
  --model-path /data-25T/models/Alpamayo-1.5-10B \
  --vlm-processor-name-or-path Qwen/Qwen3-VL-8B-Instruct \
  --output-dir outputs/dummy_training \
  --max-steps 10 \
  --batch-size 1 \
  --expert-context-mode input_only
```

切到 expert 看完整 VLM KV cache：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch \
  --config_file configs/accelerate/fsdp_4gpu_dummy.yaml \
  scripts/train_dummy_fsdp.py \
  --model-path /data-25T/models/Alpamayo-1.5-10B \
  --vlm-processor-name-or-path Qwen/Qwen3-VL-8B-Instruct \
  --output-dir outputs/dummy_training_full_kv \
  --max-steps 10 \
  --batch-size 1 \
  --expert-context-mode full_kv
```

## 说明

- 这套代码不依赖真实数据，只用 dummy dataset。
- 代码不改已有推理逻辑，只额外增加训练模块。
- 默认使用 `sdpa` attention，若环境稳定也可以传 `--attn-implementation flash_attention_2`。
