# Alpamayo 1.5 - 环境配置与运行指南

## 1. 环境信息

- **Python**: 3.12
- **CUDA**: 12.x (服务器为 CUDA 12.0, pip 安装的 PyTorch 自带 CUDA 12.4 runtime)
- **GPU**: NVIDIA H20 (96GB VRAM) x4 (单卡即可运行)
- **环境路径**: `/data/envs/alpamayo`
- **模型路径**: `/data-25T/models/Alpamayo-1.5-10B`
- **代码路径**: `/home/ubuntu/xwk/learn/alpamayo1.5`

## 2. 环境创建

```bash
# 如果在中国需要设置代理
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=http://127.0.0.1:7890

# 创建 conda 环境 (需要先将 ~/.condarc 配置为清华镜像源)
mamba create -y -p /data/envs/alpamayo python=3.12
```

**注意**: 如果 mamba/conda 无法通过代理下载, 需要临时修改 `~/.condarc` 为国内镜像:

```yaml
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

创建完环境后记得恢复原来的 `.condarc`.

## 3. 安装依赖

```bash
# 激活环境
source activate /data/envs/alpamayo
# 或者直接用绝对路径: /data/envs/alpamayo/bin/pip

# 安装 PyTorch 2.8.0 (从 PyPI, 自带 CUDA 12.4)
pip install torch==2.8.0 torchvision -i https://pypi.org/simple

# 安装其它依赖
pip install -i https://pypi.org/simple \
  "transformers==4.57.1" \
  "accelerate>=1.12.0" \
  "av>=16.0.1" \
  "einops>=0.8.1" \
  "hydra-colorlog>=1.2.0" \
  "hydra-core>=1.3.2" \
  "pandas>=2.3.3" \
  "scipy" \
  "matplotlib>=3.10.7" \
  "seaborn>=0.13.2" \
  "physical-ai-av==0.2.0"

# 安装 flash-attn (需要编译, 约 5-10 分钟)
pip install flash-attn --no-build-isolation -i https://pypi.org/simple
```

### 已安装版本参考

| 包 | 版本 |
|---|---|
| torch | 2.8.0 |
| transformers | 4.57.1 |
| flash-attn | 2.8.3 |
| accelerate | 1.13.0 |
| einops | 0.8.2 |
| physical-ai-av | 0.2.0 |

## 4. 运行推理

### 4.1 使用 dummy data 运行 (无需下载数据集)

```bash
cd /home/ubuntu/xwk/learn/alpamayo1.5

# 设置代理 (模型初始化时需从 HuggingFace 下载 Qwen3-VL-8B-Instruct 的 processor)
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=http://127.0.0.1:7890

# 指定单卡
export CUDA_VISIBLE_DEVICES=0

# 运行
/data/envs/alpamayo/bin/python run_dummy_inference.py
```

预期输出:
```
Creating dummy data...
  image_frames: torch.Size([4, 4, 3, 480, 640])
  camera_indices: tensor([0, 1, 2, 6])
  ego_history_xyz: torch.Size([1, 1, 16, 3])

Loading model from /data-25T/models/Alpamayo-1.5-10B...
Loading checkpoint shards: 100%|...| 5/5
Model loaded.
Processor ready.

=== VQA Mode ===
Answer: ['No objects satisfy the requirement.']

=== Trajectory Prediction Mode ===
pred_xyz shape: torch.Size([1, 1, 1, 64, 3])
pred_rot shape: torch.Size([1, 1, 1, 64, 3, 3])
Chain-of-Causation:
[['Keep lane since the path ahead is clear']]

Done!
```

### 4.2 使用真实数据运行 (需 HuggingFace 认证 + 数据集访问权限)

如果有 HuggingFace 账号且已获得以下 gated repo 的访问权限:
- `nvidia/PhysicalAI-Autonomous-Vehicles` (数据集)
- `nvidia/Cosmos-Reason2-8B` (原始 VLM backbone)

则可以运行官方测试脚本:

```bash
# 先登录 HuggingFace
/data/envs/alpamayo/bin/huggingface-cli login

# 运行官方推理脚本 (会自动从 HF 下载一小段示例数据)
/data/envs/alpamayo/bin/python src/alpamayo1_5/test_inference.py
```

## 5. 关键技术点

### 5.1 绕过 gated repo 问题

模型 config 中 `vlm_name_or_path` 设为 `nvidia/Cosmos-Reason2-8B` (gated repo). 初始化时代码会尝试从该 repo 下载 processor 和 VLM config. 

**解决方案**: 在 `run_dummy_inference.py` 中, 手动加载 `config.json`, 将 `vlm_name_or_path` 替换为 `Qwen/Qwen3-VL-8B-Instruct` (公开可访问, 架构相同).

### 5.2 Attention 实现选择

- **`flash_attention_2`** (默认): VLM 推理可用, 但在 expert diffusion model 中会报错 (`cu_seqlens_q must have shape (batch_size + 1)`), 这是因为 expert 使用了自定义的 4D attention mask, 与当前 flash-attn 2.8.3 + transformers 4.57.1 不兼容.
- **`sdpa`** (推荐): PyTorch 内置 scaled dot-product attention, VLM 和 expert 均可正常工作. 性能略低于 flash-attn 但功能完整.

脚本中已设置为 `sdpa`.

### 5.3 模型架构

Alpamayo 1.5 由两部分组成:
1. **VLM (Vision-Language Model)**: 基于 Qwen3-VL-8B, 处理多摄像头图像 + 历史轨迹, 生成 Chain-of-Causation 推理文本
2. **Expert Diffusion Model**: 基于 VLM 的 hidden states, 通过 flow matching 扩散过程生成未来 64 步 (6.4秒) 的轨迹预测

### 5.4 两种推理模式

1. **轨迹预测** (`sample_trajectories_from_data_with_vlm_rollout`): 完整 pipeline, 输入多摄像头图像 + 历史轨迹, 输出推理文本 + 未来轨迹
2. **VQA** (`generate_text`): 纯文本生成, 输入图像 + 问题, 输出文字回答

### 5.5 VRAM 使用

| 配置 | 显存 |
|---|---|
| 单样本 (`num_traj_samples=1`) | ~24 GB |
| 多样本 (`num_traj_samples=16`) | ~40 GB |
| 多样本 + CFG (`num_traj_samples=16`) | ~60 GB |

H20 (96GB) 单卡足够运行所有配置.

## 6. 文件说明

| 文件 | 说明 |
|---|---|
| `run_dummy_inference.py` | 使用合成数据的推理脚本 (本次新增) |
| `src/alpamayo1_5/test_inference.py` | 官方推理脚本 (需 HF 数据集) |
| `src/alpamayo1_5/helper.py` | 消息构建、processor 获取等工具函数 |
| `src/alpamayo1_5/models/alpamayo1_5.py` | 模型主类定义 |
| `src/alpamayo1_5/models/base_model.py` | 基础 VLA 模型类 |
| `notebooks/` | 各种交互式推理 notebook |
