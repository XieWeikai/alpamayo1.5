# Alpamayo-1.5-10B 模型架构详解

## 1. 总览

Alpamayo-1.5-10B 是 NVIDIA 提出的一个 **Vision-Language-Action (VLA)** 模型，用于自动驾驶场景下的因果推理 (Chain-of-Causation, CoC) 和轨迹预测。模型由三个核心部分组成：

| 组件 | 作用 | 实现 |
|------|------|------|
| **VLM 主干** | 多摄像头图像理解 + 文本推理生成 | Qwen3-VL-8B-Instruct |
| **Expert 去噪器** | 基于 VLM 上下文，预测动作向量场 | Qwen3 Text-only Transformer (无 embed_tokens) |
| **Flow Matching 扩散** | 通过 Euler 积分从噪声中采样动作序列 | FlowMatching (Euler, 10 步) |

最终输出：**未来 6.4 秒的驾驶轨迹** (64 个路点, 10Hz) + **CoC 推理文本**。

> **符号约定**：本文档中 `B` 表示 batch size，`ns` = `num_traj_sets` (轨迹集合数，默认 1)，`nj` = `num_traj_samples` (每集合的采样数，如 6)。中间计算过程的 batch 维度通常为 `B * ns * nj`（因为每条输入会产生多条采样），文中有时简写为 `B*ns`（此处 ns 代表 `n_samples_total = num_traj_sets * num_traj_samples`，即总采样数）。最终输出会 reshape 回 `(B, ns, nj, ...)`。

---

## 2. 整体架构图

```
                              ┌─────────────────────────────────┐
                              │          原始输入                │
                              │  image_frames (N_cam, F, 3,H,W) │
                              │  ego_history_xyz  (B,1,16,3)    │
                              │  ego_history_rot  (B,1,16,3,3)  │
                              │  [可选] navigation text          │
                              └───────────────┬─────────────────┘
                                              │
                        ┌─────────────────────┼──────────────────────┐
                        ▼                     ▼                      ▼
                ┌──────────────┐    ┌─────────────────┐    ┌─────────────────┐
                │ 图像预处理    │    │ 轨迹历史离散化   │    │ 文本 Prompt 构建 │
                │ Qwen3-VL     │    │ DeltaTrajectory  │    │ create_message() │
                │ Processor    │    │ Tokenizer        │    │ helper.py:77     │
                └──────┬───────┘    └────────┬─────────┘    └────────┬────────┘
                       │                     │                       │
                       └─────────────────────┼───────────────────────┘
                                             ▼
                              ┌──────────────────────────────┐
                              │     Token 融合 & Tokenize     │
                              │  fuse_traj_tokens()           │
                              │  base_model.py:172            │
                              │  input_ids: (B, L)            │
                              └──────────────┬───────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────┐
                              │   VLM 自回归生成 (Qwen3-VL)   │
                              │   vlm.generate()              │
                              │   alpamayo1_5.py:285          │
                              │                               │
                              │   输出:                        │
                              │   sequences (B*ns, L')        │
                              │   past_key_values (KV Cache)  │
                              │   rope_deltas                 │
                              └──────────────┬───────────────┘
                                             │
                        ┌────────────────────┼────────────────────┐
                        ▼                                        ▼
             ┌────────────────────┐               ┌──────────────────────────┐
             │ 文本提取            │               │ 构建 Expert 输入          │
             │ extract_text_tokens │               │ _build_expert_pos_ids_   │
             │ token_utils.py:151  │               │  and_attn_mask()         │
             │                    │               │ alpamayo1_5.py:158       │
             │ cot / meta_action  │               │                          │
             │ / answer           │               │ position_ids: (3,B*ns,64)│
             └────────────────────┘               │ attn_mask: (B*ns,1,64,KV)│
                                                  └────────────┬─────────────┘
                                                               │
                                                               ▼
                                            ┌─────────────────────────────────┐
                                            │    Flow Matching 扩散采样循环    │
                                            │    flow_matching.py:138          │
                                            │                                 │
                                            │  x₀ ~ N(0,I), shape (B*ns,64,2)│
                                            │                                 │
                                            │  for t in [0, 1]:               │
                                            │    ┌──────────────────────────┐  │
                                            │    │ action_in_proj(x, t)     │  │
                                            │    │ → (B*ns, 64, hidden)     │  │
                                            │    ├──────────────────────────┤  │
                                            │    │ Expert Transformer       │  │
                                            │    │ + VLM KV Cache           │  │
                                            │    │ → (B*ns, 64, hidden)     │  │
                                            │    ├──────────────────────────┤  │
                                            │    │ action_out_proj          │  │
                                            │    │ → v(x,t): (B*ns, 64, 2) │  │
                                            │    └──────────────────────────┘  │
                                            │    x ← x + dt * v(x, t)        │
                                            │                                 │
                                            │  输出: sampled_action (B*ns,64,2)│
                                            └───────────────┬─────────────────┘
                                                            │
                                                            ▼
                                            ┌─────────────────────────────────┐
                                            │   动作空间 → 轨迹转换            │
                                            │   UnicycleAccelCurvature        │
                                            │   .action_to_traj()             │
                                            │   unicycle_accel_curvature.py:300│
                                            │                                 │
                                            │   [accel, kappa] → (x, y, yaw)  │
                                            │                                 │
                                            │   pred_xyz: (B, ns, 64, 3)      │
                                            │   pred_rot: (B, ns, 64, 3, 3)   │
                                            └─────────────────────────────────┘
```

---

## 3. 模型类继承关系

```
PreTrainedModel (HuggingFace)
  └── ReasoningVLA  (base_model.py:289)    ← 同时混入 TrajectoryFusionMixin
        ├── vlm: Qwen3VLForConditionalGeneration   ← VLM 主干
        ├── hist_traj_tokenizer: DeltaTrajectoryTokenizer
        └── tokenizer: AutoTokenizer (扩展词汇表)
              │
              └── Alpamayo1_5  (alpamayo1_5.py:82)
                    ├── expert: Qwen3 TextModel (无 embed_tokens)
                    ├── action_space: UnicycleAccelCurvatureActionSpace
                    ├── diffusion: FlowMatching
                    ├── action_in_proj: PerWaypointActionInProjV2
                    └── action_out_proj: nn.Linear
```

配置类:
```
PretrainedConfig
  └── ReasoningVLAConfig  (base_model.py:204)
        └── Alpamayo1_5Config  (config.py:23)
```

---

## 4. 逐步数据流详解

### 4.1 数据加载

**代码位置**: `load_physical_aiavdataset.py:27-219`

从 PhysicalAI-AV 数据集加载多摄像头视频帧和自车轨迹数据。

**输出张量**:

| 张量 | Shape | 含义 |
|------|-------|------|
| `image_frames` | `(N_cam, num_frames, 3, H, W)` | 多摄像头 RGB 图像，默认 N_cam=4, num_frames=4。H, W 由 PhysicalAI-AV 数据集中各摄像头原始分辨率决定 (代码中未硬编码)，但后续 Qwen3-VL Processor 会自动 resize 到 `min_pixels=163840` ~ `max_pixels=196608` 的等效范围 |
| `camera_indices` | `(N_cam,)` | 摄像头类型索引，0-6 对应 7 个摄像头位置 |
| `ego_history_xyz` | `(1, 1, 16, 3)` | 过去 1.6s 自车位置 (局部坐标系, 10Hz) |
| `ego_history_rot` | `(1, 1, 16, 3, 3)` | 过去 1.6s 自车旋转矩阵 (SO(3)) |
| `ego_future_xyz` | `(1, 1, 64, 3)` | 未来 6.4s 自车位置 (ground truth) |
| `ego_future_rot` | `(1, 1, 64, 3, 3)` | 未来 6.4s 自车旋转矩阵 (ground truth) |

**摄像头索引映射** (`load_physical_aiavdataset.py:9-17`):

| Index | 摄像头名称 | FOV |
|-------|-----------|-----|
| 0 | camera_cross_left_120fov | 120 |
| 1 | camera_front_wide_120fov | 120 |
| 2 | camera_cross_right_120fov | 120 |
| 3 | camera_rear_left_70fov | 70 |
| 4 | camera_rear_tele_30fov | 30 |
| 5 | camera_rear_right_70fov | 70 |
| 6 | camera_front_tele_30fov | 30 |

---

### 4.2 Chat Message 构建

**代码位置**: `helper.py:77-142` (`create_message()`)

将图像帧和文本 prompt 组织成 Qwen3-VL 所需的对话格式。

**消息结构**:

```python
[
  # System prompt
  {"role": "system", "content": "You are a driving assistant..."},

  # User message: 图像 + 轨迹历史 token + 导航文本 + 指令
  {"role": "user", "content": [
      # 图像内容 (每个摄像头的每一帧)
      {"type": "text", "text": "Front camera: "},
      {"type": "text", "text": "frame 0 "},
      {"type": "image", "image": tensor(3, H, W)},  # 逐帧添加
      ...
      # 文本内容
      {"type": "text", "text":
        "<|traj_history_start|>"
        "<|traj_history|>" * 48    # 48 个占位符 token
        "<|traj_history_end|>"
        "<|route_start|>{nav_text}<|route_end|>"  # 可选导航
        "output the chain-of-thought reasoning..."
      }
  ]},

  # Assistant prefix (强制模型从 CoC 开始生成)
  {"role": "assistant", "content": "<|cot_start|>"}
]
```

**关键参数**:
- `num_traj_token = 48`: 轨迹历史占位符数量 (`helper.py:106`)
- 每个占位符将在后续步骤被替换为实际的离散轨迹 token

---

### 4.3 Processor Tokenize

**代码位置**: `helper.py:190-199` (`get_processor()`)，`test_inference.py:42-49`

使用 Qwen3-VL Processor 将对话消息转为模型输入。

```python
processor = helper.get_processor(model.tokenizer)
inputs = processor.apply_chat_template(messages, tokenize=True, ...)
```

**Processor 配置**:
- `min_pixels = 163840` (约 405x405)
- `max_pixels = 196608` (约 443x443)
- 基础 Processor: `Qwen/Qwen3-VL-2B-Instruct` (仅使用其 image processor)
- Tokenizer 来自模型本身 (已扩展词汇表)

**输出张量**:

| 张量 | Shape | 含义 |
|------|-------|------|
| `input_ids` | `(B, L)` | token 序列，包含文本 token + 图像占位 token + 轨迹占位 token |
| `attention_mask` | `(B, L)` | 注意力掩码，1=有效, 0=padding |
| `pixel_values` | `(N_patches, 3, patch_h, patch_w)` | Qwen3-VL 处理后的图像 patch |
| `image_grid_thw` | `(N_images, 3)` | 每张图像的 (temporal, height, width) 网格尺寸 |

---

### 4.4 词汇表扩展

**代码位置**: `base_model.py:255-286` (`_build_processor()`)

在标准 Qwen3 词汇表基础上添加:

1. **768 个离散轨迹 token**: `<i0>` 到 `<i767>` (`base_model.py:268-274`)
   - `traj_token_start_idx`: `<i0>` 在词汇表中的起始 ID
   - 用于编码历史轨迹的离散化表示

2. **特殊 token** (`base_model.py:48-79`):

| Token | 含义 |
|-------|------|
| `<\|traj_history_start\|>` | 轨迹历史序列开始 |
| `<\|traj_history\|>` | 轨迹历史占位符 (会被替换) |
| `<\|traj_history_end\|>` | 轨迹历史序列结束 |
| `<\|cot_start\|>` | Chain-of-Thought 推理开始 |
| `<\|cot_end\|>` | Chain-of-Thought 推理结束 |
| `<\|traj_future_start\|>` | 未来轨迹开始 (VLM 生成的 EOS) |
| `<\|traj_future_end\|>` | 未来轨迹结束 |
| `<\|route_start\|>` / `<\|route_end\|>` | 导航指令边界 |
| `<\|question_start\|>` / `<\|answer_start\|>` | VQA 问答边界 |

---

### 4.5 轨迹历史 Token 化

**代码位置**: `base_model.py:95-126` (`tokenize_history_trajectory()`)，`delta_tokenizer.py:21-97`

将连续的自车历史轨迹离散化为 token。

**流程**:

```
ego_history_xyz: (B, 1, 16, 3)   ─┐
ego_history_rot: (B, 1, 16, 3, 3) ─┤
                                    ▼
                    DeltaTrajectoryTokenizer.encode()
                    (delta_tokenizer.py:47-97)
                                    │
                                    ▼
                    hist_idx: (B, 48)  ← 16步 × 3维(xyz) = 48 token
```

**DeltaTrajectoryTokenizer 编码过程** (`delta_tokenizer.py:47-97`):

1. **计算 delta**: `delta_xyz[t] = xyz[t+1] - xyz[t]`, shape `(B, T, 3)`
2. **归一化**: 将 delta 映射到 `[0, 1]`，范围由 `ego_xyz_min=(-4,-4,-10)`, `ego_xyz_max=(4,4,10)` 定义
3. **量化**: `token = round(normalized * 999)`，映射到 `[0, 999]` 共 1000 个 bin (`num_bins=1000`)
4. **展平**: `(B, T, 3)` → `(B, T*3)` = `(B, 48)`
5. **加偏移**: `hist_idx += traj_token_start_idx`，映射到词汇表中的离散轨迹 token ID

---

### 4.6 Token 融合

**代码位置**: `base_model.py:172-201` (`fuse_traj_tokens()`)

将 `input_ids` 中的 `<|traj_history|>` 占位符替换为实际的轨迹 token ID。

```python
# input_ids 中有 48 个 <|traj_history|> 占位符
# hist_idx 包含 48 个实际的离散轨迹 token ID
input_ids = replace_pad_token(input_ids, hist_idx, traj_token_ids["history"])
```

**核心函数** `replace_pad_token()` (`base_model.py:89-92`):
- 找到 `input_ids` 中所有值为 `traj_token_ids["history"]` 的位置
- 用 `hist_idx` 中的实际 token ID 依次替换

**融合后的 `input_ids` 结构**:

```
[system_tokens] [image_tokens...] [traj_history_start] [i23][i156][i789]...(48个) [traj_history_end] [route_start]...[route_end] [prompt_tokens] [cot_start]
```

---

### 4.7 VLM 自回归生成

**代码位置**: `alpamayo1_5.py:258-300`

这是推理流程的第一个核心阶段：VLM 处理所有输入（图像 + 轨迹历史 + 文本），生成 CoC 推理文本，直到遇到 `<|traj_future_start|>` token。

**生成配置** (`alpamayo1_5.py:262-271`):

```python
generation_config.top_p = 0.98
generation_config.temperature = 0.6
generation_config.do_sample = True
generation_config.num_return_sequences = num_traj_samples  # 如 6
generation_config.max_new_tokens = 256  # 最大生成长度
```

**关键机制**:

1. **ExpertLogitsProcessor** (`alpamayo1_5.py:48-79`):
   - 在 VLM 生成时，将离散轨迹 token 的 logits 设为 `-inf`
   - 防止 VLM 生成轨迹 token，确保只输出文本推理

2. **StopAfterEOS** (`token_utils.py:172-209`):
   - 自定义停止条件：在生成 `<|traj_future_start|>` token 后再多生成一个 token 再停止
   - 多生成一个的原因：KV Cache 需要在下一个 token 生成后才更新到完整状态

```python
eos_token_id = tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
vlm_outputs = self.vlm.generate(
    input_ids=input_ids,        # (B, L)
    pixel_values=...,           # 图像 patch
    image_grid_thw=...,         # 图像网格信息
    generation_config=...,
    stopping_criteria=[StopAfterEOS(eos_token_id)],
    logits_processor=[ExpertLogitsProcessor(...)],
)
```

**VLM 生成的 token 序列示例**:

```
...[cot_start] The vehicle ahead is slowing down. The ego vehicle should
maintain safe distance and prepare to brake. [cot_end][traj_future_start] <extra_token>
```

**输出张量**:

| 张量 | Shape | 含义 |
|------|-------|------|
| `vlm_outputs.sequences` | `(B*num_traj_samples, L')` | 完整生成的 token 序列 |
| `vlm_outputs.past_key_values` | KV Cache | 所有已生成 token 的 Key/Value 缓存，供 Expert 复用 |
| `vlm_outputs.rope_deltas` | `(B*num_traj_samples, 1)` | VLM 的 RoPE 位置偏移量 |

其中 `L'` 是生成后的完整序列长度，`B*num_traj_samples` 是因为每个输入生成 `num_traj_samples` 个不同的推理序列。

**padding 后处理** (`alpamayo1_5.py:294-299`):
```python
# 将 <|traj_future_start|> 之后的 token 替换为 pad_token
vlm_outputs.sequences = replace_padding_after_eos(...)
```

---

### 4.8 构建 Expert 位置编码和注意力掩码

**代码位置**: `alpamayo1_5.py:158-212` (`_build_expert_pos_ids_and_attn_mask()`)

Expert Transformer 需要处理 64 个扩散 token (对应 64 个未来路点)。它复用 VLM 的 KV Cache 作为前缀上下文，因此需要精心构建位置编码和注意力掩码。

**Step 1: 找到 EOS 偏移** (`alpamayo1_5.py:303-309`)

```python
offset = self._find_eos_offset(sequences, eos_token_id, device)
# offset: (B*ns,), 每个序列中 <|traj_future_start|> 的位置 + 1
```

**Step 2: 构建 3D RoPE 位置编码** (`alpamayo1_5.py:186-189`)

Qwen3-VL 使用 3 分量 RoPE: (temporal, height, width)。Expert 的扩散 token 需要延续 VLM 的位置编码。

```python
position_ids = torch.arange(n_diffusion_tokens)  # [0, 1, ..., 63]
position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star)  # (3, B*ns, 64)
position_ids += (rope_deltas + offset[:, None])  # 加上 VLM 的位置偏移
```

| 张量 | Shape | 含义 |
|------|-------|------|
| `position_ids` | `(3, B*ns, 64)` | 3 维 RoPE 位置 ID，分别对应 temporal/height/width |

**Step 3: 构建 4D 注意力掩码** (`alpamayo1_5.py:192-210`)

```python
attention_mask = torch.zeros(
    (b_star, 1, n_diffusion_tokens, kv_cache_seq_len + n_diffusion_tokens)
)  # shape: (B*ns, 1, 64, KV+64)
```

掩码作用:
- 扩散 token 可以 attend to VLM KV Cache 中的所有有效 token
- 掩盖 `offset` 和扩散 token 之间的空隙 (这些位置没有实际 token)
- 如果输入有 left-padding，也将 padding 位置掩盖

| 张量 | Shape | 含义 |
|------|-------|------|
| `attention_mask` | `(B*ns, 1, 64, KV+64)` | 4D 注意力掩码, 0=attend, -inf=masked |

---

### 4.9 扩散采样循环 (Flow Matching)

**代码位置**: `flow_matching.py:138-196` (`_euler()`), `alpamayo1_5.py:327-370`

这是推理的核心计算阶段。通过 Flow Matching 的 Euler 积分，从高斯噪声中逐步采样出动作序列。

**初始化** (`flow_matching.py:171`):

```python
x = torch.randn(batch_size, *self.x_dims) * temperature
# x shape: (B*ns, 64, 2), 其中 x_dims = (64, 2)
# 64 = n_waypoints, 2 = (acceleration, curvature)
```

**时间步** (`flow_matching.py:172`):

```python
time_steps = torch.linspace(0.0, 1.0, num_inference_steps + 1)
# 默认 num_inference_steps=10, 生成 11 个时间点: [0.0, 0.1, ..., 1.0]
```

**Euler 积分循环** (`flow_matching.py:177-191`):

```python
for i in range(num_inference_steps):
    dt = time_steps[i+1] - time_steps[i]     # 步长 0.1
    t_start = time_steps[i]                   # 当前时间
    v = step_fn(x=x, t=t_start)              # 预测向量场 v(x, t)
    x = x + dt * v                            # Euler 更新: x ← x + dt * v(x, t)
```

每一步的 `step_fn` 内部执行以下三个子步骤：

#### 4.9.1 Action Input Projection

**代码位置**: `action_in_proj.py:104-167` (`PerWaypointActionInProjV2`)

将带噪声的动作 `x` 和时间步 `t` 编码为 Expert Transformer 的输入嵌入。

**架构**:

```
x: (B*ns, 64, 2)          t: (B*ns, 1, 1)
     │                          │
     ▼                          ▼
 FourierEncoderV2 ×2      FourierEncoderV2
 (每个动作维度一个)          (时间步编码)
     │                          │
     ▼                          ▼
 action_feats              timestep_feats
 (B*ns, 64, 40)            (B*ns, 64, 20)
     │                          │
     └──────────┬───────────────┘
                ▼
         cat → (B*ns, 64, 60)
                │
                ▼
          MLPEncoder (4层)
          Linear(60→1024) → SiLU
          RMSNorm → Linear(1024→1024) → SiLU  ×3
          RMSNorm → Linear(1024→hidden_size)
                │
                ▼
          LayerNorm
                │
                ▼
         (B*ns, 64, hidden_size)
```

**FourierEncoderV2** (`action_in_proj.py:73-101`):

对标量输入 `x` 进行傅里叶特征编码：

```python
freqs = torch.logspace(0, log10(100), steps=half_dim)  # 对数间隔频率 [1, ..., 100]
arg = x[..., None] * freqs * 2π                         # (*, half_dim)
output = [sin(arg), cos(arg)] * sqrt(2)                  # (*, dim)
```

- 默认 `dim=20` → `half_dim=10`, 10 个频率
- 每个动作维度 (accel, kappa) 各自编码 → 共 20+20=40 维
- 时间步编码 → 20 维
- 拼接后 → 60 维输入 MLP

**张量变化**:

| 步骤 | Shape | 含义 |
|------|-------|------|
| 输入 x | `(B*ns, 64, 2)` | 带噪声的归一化动作 [accel, kappa] |
| 输入 t | `(B*ns, 1, 1)` | 扩散时间步 ∈ [0, 1] |
| FourierEncoderV2(accel) | `(B*ns, 64, 20)` | 加速度的傅里叶特征 |
| FourierEncoderV2(kappa) | `(B*ns, 64, 20)` | 曲率的傅里叶特征 |
| FourierEncoderV2(t) | `(B*ns, 64, 20)` | 时间步的傅里叶特征 (repeat 到 64) |
| cat | `(B*ns, 64, 60)` | 拼接所有傅里叶特征 |
| MLPEncoder 输出 | `(B*ns, 64, hidden_size)` | Expert 输入嵌入 |

#### 4.9.2 Expert Transformer Forward

**代码位置**: `alpamayo1_5.py:342-356`

Expert 是一个 **无 embed_tokens** 的 Qwen3 Text Transformer (`alpamayo1_5.py:97-103`)。它直接接收嵌入向量作为输入，不经过词嵌入层。

```python
expert_config = copy.deepcopy(self.vlm.config.text_config)
self.expert = AutoModel.from_config(expert_config)
del self.expert.embed_tokens   # 删除词嵌入层
```

**关键特性**: Expert 使用 **非因果注意力** (`is_causal=False`, `alpamayo1_5.py:324-325`)，即 64 个扩散 token 之间可以互相 attend。

**前向传播**:

```python
expert_out = self.expert(
    inputs_embeds=future_token_embeds,  # (B*ns, 64, hidden_size) ← 来自 action_in_proj
    position_ids=position_ids,          # (3, B*ns, 64) ← 3D RoPE
    past_key_values=prompt_cache,       # VLM 生成的 KV Cache
    attention_mask=attention_mask,      # (B*ns, 1, 64, KV+64)
    use_cache=True,
    is_causal=False,                    # 非因果注意力
)
```

Expert 通过 `past_key_values` 可以 attend to VLM 之前处理过的所有 token (图像、轨迹历史、CoC 推理文本)，从而隐式地以推理结果为条件进行轨迹预测。

**每次前向后裁剪 KV Cache** (`alpamayo1_5.py:351`):
```python
prompt_cache.crop(prefill_seq_len)
# 删除本次前向中新增的扩散 token 的 KV，恢复到仅含 VLM 前缀的状态
# 这样下一个扩散步可以复用同一份 KV Cache
```

**张量变化**:

| 步骤 | Shape | 含义 |
|------|-------|------|
| 输入 inputs_embeds | `(B*ns, 64, hidden_size)` | 傅里叶编码后的动作嵌入 |
| Expert 输出 last_hidden_state | `(B*ns, 64, hidden_size)` | 上下文感知的隐状态 |

#### 4.9.3 Action Output Projection

**代码位置**: `alpamayo1_5.py:116-119`, `354-356`

一个简单的线性层，将 Expert 隐状态映射回动作空间:

```python
self.action_out_proj = nn.Linear(expert_hidden_size, 2)
# expert_hidden_size → 2 (acceleration, curvature)
```

```python
last_hidden = expert_out.last_hidden_state[:, -64:]   # (B*ns, 64, hidden_size)
pred = self.action_out_proj(last_hidden)                # (B*ns, 64, 2)
pred = pred.view(-1, 64, 2)                             # 向量场 v(x, t)
```

| 步骤 | Shape | 含义 |
|------|-------|------|
| 输入 | `(B*ns, 64, hidden_size)` | Expert 隐状态 |
| 输出 | `(B*ns, 64, 2)` | 预测的向量场 v(x, t)，用于 Euler 更新 |

**扩散循环结束后**:

```python
sampled_action  # shape: (B*ns, 64, 2), 去噪后的归一化动作序列
```

---

### 4.10 动作空间到轨迹的转换

**代码位置**: `unicycle_accel_curvature.py:300-382` (`action_to_traj()`)

使用 **单车运动学模型 (Unicycle Kinematic Model)** 将归一化的 `[accel, kappa]` 动作序列转换为世界坐标系下的轨迹。

**Step 1: 反归一化** (`unicycle_accel_curvature.py:319-326`):

```python
accel = action[..., 0] * accel_std + accel_mean   # 真实加速度 (m/s²)
kappa = action[..., 1] * curvature_std + curvature_mean  # 真实曲率 (1/m)
```

**Step 2: 估计初始速度** (`unicycle_accel_curvature.py:328-331`):

```python
t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)
v0 = t0_states["v"]  # 从历史轨迹估计 t=0 时刻的速度
```

`estimate_t0_states()` (`unicycle_accel_curvature.py:207-222`):
- 对历史轨迹做 Tikhonov 正则化求解速度
- 返回最后一个时间步的速度作为 v0

**Step 3: 运动学积分** (`unicycle_accel_curvature.py:334-366`):

```python
dt = 0.1  # 时间步长, 10Hz

# 速度: v(t) = v0 + ∑(accel * dt)
velocity = cat([v0, v0 + cumsum(accel * dt)])     # (..., N+1)

# 航向角: θ(t) = θ0 + ∑(kappa * v * dt + kappa * accel * dt²/2)
theta = cat([0, cumsum(kappa * v[:-1] * dt) + cumsum(kappa * accel * dt²/2)])  # (..., N+1)

# 位置: 梯形积分
#   x(t) = ∑[ (v[t]*cos(θ[t]) + v[t+1]*cos(θ[t+1])) * dt/2 ]
#   y(t) = ∑[ (v[t]*sin(θ[t]) + v[t+1]*sin(θ[t+1])) * dt/2 ]
x = cumsum(v[:-1]*cos(θ[:-1]) * dt/2) + cumsum(v[1:]*cos(θ[1:]) * dt/2)  # (..., N)
y = cumsum(v[:-1]*sin(θ[:-1]) * dt/2) + cumsum(v[1:]*sin(θ[1:]) * dt/2)  # (..., N)
```

**Step 4: 组装输出** (`unicycle_accel_curvature.py:367-382`):

```python
traj_future_xyz = zeros(..., 64, 3)
traj_future_xyz[..., 0] = x       # X 坐标
traj_future_xyz[..., 1] = y       # Y 坐标
traj_future_xyz[..., 2] = z_hist  # Z 坐标继承自历史最后一步

traj_future_rot = rot_2d_to_3d(rotation_matrix_torch(theta[..., 1:]))  # 从 yaw 构建 SO(3) 旋转矩阵
```

**张量变化**:

| 步骤 | Shape | 含义 |
|------|-------|------|
| 输入 action | `(B*ns, 64, 2)` | 归一化 [accel, kappa] |
| accel (反归一化) | `(B*ns, 64)` | 加速度 (m/s²)，范围 [-9.8, 9.8] |
| kappa (反归一化) | `(B*ns, 64)` | 曲率 (1/m)，范围 [-0.2, 0.2] |
| velocity | `(B*ns, 65)` | 速度序列 (N+1 个时间点) |
| theta | `(B*ns, 65)` | 航向角序列 (N+1 个时间点) |
| x, y | `(B*ns, 64)` | 局部坐标系下的 XY 位置 |
| traj_future_xyz | `(B*ns, 64, 3)` | 未来轨迹 XYZ |
| traj_future_rot | `(B*ns, 64, 3, 3)` | 未来轨迹旋转矩阵 |

---

### 4.11 输出重塑

**代码位置**: `alpamayo1_5.py:384-390`

```python
pred_xyz = rearrange(pred_xyz, "(b ns nj) ... -> b ns nj ...",
                     ns=num_traj_sets, nj=num_traj_samples)
pred_rot = rearrange(pred_rot, "(b ns nj) ... -> b ns nj ...",
                     ns=num_traj_sets, nj=num_traj_samples)
```

**最终输出张量**:

| 张量 | Shape | 含义 |
|------|-------|------|
| `pred_xyz` | `(B, num_traj_sets, num_traj_samples, 64, 3)` | 预测轨迹位置 |
| `pred_rot` | `(B, num_traj_sets, num_traj_samples, 64, 3, 3)` | 预测轨迹旋转矩阵 |

- `B`: batch size
- `num_traj_sets`: 轨迹集合数 (默认 1)
- `num_traj_samples`: 每组轨迹的采样数 (如 6)
- `64`: 未来路点数 (6.4s @ 10Hz)
- `3` / `3x3`: 位置 XYZ / SO(3) 旋转矩阵

---

### 4.12 文本输出提取

**代码位置**: `token_utils.py:151-169` (`extract_text_tokens()`)

从 VLM 生成的 token 序列中提取结构化文本:

```python
extra = extract_text_tokens(tokenizer, vlm_outputs.sequences)
# extra = {
#     "cot": ["The vehicle ahead is decelerating...", ...],       # Chain-of-Thought
#     "meta_action": ["brake gently, maintain lane", ...],         # 动作描述
#     "answer": ["", ...],                                         # VQA 回答 (如有)
# }
```

**提取逻辑** (`token_utils.py:123-148` `extract_between_special_tokens()`):

对每个文本字段，找到 `<|{field}_start|>` 和 `<|{field}_end|>` 之间的内容。

---

## 5. 导航条件推理 (Classifier-Free Guidance)

**代码位置**: `alpamayo1_5.py:404-691` (`sample_trajectories_from_data_with_vlm_rollout_cfg_nav()`)

当提供导航指令时，可以使用 CFG 来增强导航条件的影响。

**核心思想**: 同时维护两份 KV Cache — 有导航和无导航 — 在扩散采样时融合两者的预测。

**流程差异**:

1. **有导航 KV Cache**: 正常 VLM 生成 (包含 `<|route_start|>nav_text<|route_end|>`)
2. **无导航 KV Cache**: 移除导航文本后重新 prefill (`alpamayo1_5.py:517-596`)
   ```python
   unguided_input_ids = remove_nav_text(input_ids, tokenizer)
   unguided_prefill_outputs = self.vlm(input_ids=unguided_input_ids, ...)
   ```
3. **扩散采样时使用 Guided Flow** (`flow_matching.py:114-136`):
   ```python
   v_guided = (1 - w) * v_unguided + w * v_guided
   # w = inference_guidance_weight
   ```

---

## 6. VQA (视觉问答) 模式

**代码位置**: `base_model.py:451-500` (`generate_text()`)

纯文本生成模式，不涉及扩散轨迹预测。

```python
# 构建 VQA message
messages = helper.create_vqa_message(frames, question="What color is the traffic light?")

# 调用
extra = model.generate_text(data=model_inputs, max_generation_length=256)
# extra["answer"] = ["The traffic light is red.", ...]
```

---

## 7. 配置参数总结

### 7.1 ReasoningVLAConfig (`base_model.py:204-240`)

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `vlm_name_or_path` | `"Qwen/Qwen3-VL-8B-Instruct"` | VLM 主干模型 |
| `traj_vocab_size` | `768` | 离散轨迹 token 数量 |
| `tokens_per_history_traj` | `16` | 历史轨迹步数 |
| `tokens_per_future_traj` | `64` | 未来轨迹步数 |
| `model_dtype` | `"bfloat16"` | 模型精度 |
| `attn_implementation` | `"flash_attention_2"` | 注意力实现方式 |

### 7.2 Alpamayo1_5Config (`config.py:23-50`)

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `diffusion_cfg` | FlowMatching 配置 | 扩散模型配置 |
| `action_space_cfg` | Unicycle 配置 | 动作空间配置 |
| `action_in_proj_cfg` | PerWaypointActionInProjV2 配置 | 动作输入投影配置 |
| `action_out_proj_cfg` | nn.Linear 配置 | 动作输出投影配置 |
| `expert_cfg` | Expert Transformer 覆盖配置 | 可覆盖 Expert 的层数、头数等 |
| `keep_same_dtype` | `True` | 保持所有组件 dtype 一致 |
| `expert_non_causal_attention` | `True` | Expert 使用非因果注意力 |

### 7.3 FlowMatching (`flow_matching.py:32-50`)

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `int_method` | `"euler"` | 积分方法 |
| `num_inference_steps` | `10` | 扩散推理步数 |
| `inference_guidance_weight` | `1.0` | CFG 引导权重 |

### 7.4 UnicycleAccelCurvatureActionSpace (`unicycle_accel_curvature.py:39-56`)

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `accel_bounds` | `(-9.8, 9.8)` | 加速度范围 (m/s²) |
| `curvature_bounds` | `(-0.2, 0.2)` | 曲率范围 (1/m) |
| `dt` | `0.1` | 时间步长 (s)，对应 10Hz |
| `n_waypoints` | `64` | 路点数 (6.4s) |
| `accel_mean/std` | `0.0 / 1.0` | 加速度归一化参数 |
| `curvature_mean/std` | `0.0 / 1.0` | 曲率归一化参数 |

---

## 8. 端到端 Tensor Shape 追踪表

以 `B=1, N_cam=4, F=4, num_traj_samples=6` 为例:

| 阶段 | 张量 | Shape | 说明 |
|------|------|-------|------|
| 数据加载 | image_frames | `(4, 4, 3, H, W)` | 4 摄像头, 4 帧 |
| 数据加载 | ego_history_xyz | `(1, 1, 16, 3)` | 1.6s 历史位置 |
| 数据加载 | ego_history_rot | `(1, 1, 16, 3, 3)` | 1.6s 历史旋转 |
| Tokenize | input_ids | `(1, L)` | L ≈ 数千 (含图像 token) |
| Tokenize | pixel_values | `(N_patches, 3, pH, pW)` | 图像 patch |
| Tokenize | image_grid_thw | `(N_images, 3)` | 图像网格信息 |
| 轨迹 Token 化 | hist_idx | `(1, 48)` | 16步 × 3维 = 48 token |
| Token 融合 | input_ids (融合后) | `(1, L)` | 占位符已替换为真实 ID |
| VLM 生成 | sequences | `(6, L')` | 6 条不同推理路径 |
| VLM 生成 | past_key_values | KV Cache, seq_len=L' | 供 Expert 复用 |
| VLM 生成 | rope_deltas | `(6, 1)` | RoPE 偏移 |
| Expert 准备 | offset | `(6,)` | EOS 位置 + 1 |
| Expert 准备 | position_ids | `(3, 6, 64)` | 3D RoPE |
| Expert 准备 | attention_mask | `(6, 1, 64, KV+64)` | 4D 掩码 |
| 扩散初始化 | x (噪声) | `(6, 64, 2)` | 高斯噪声 |
| action_in_proj | future_token_embeds | `(6, 64, hidden_size)` | 傅里叶+MLP 编码 |
| Expert 前向 | last_hidden_state | `(6, 64, hidden_size)` | 上下文感知隐状态 |
| action_out_proj | v(x, t) | `(6, 64, 2)` | 向量场预测 |
| 扩散输出 | sampled_action | `(6, 64, 2)` | 去噪动作序列 |
| 动作→轨迹 | pred_xyz | `(6, 64, 3)` | 局部坐标系位置 |
| 动作→轨迹 | pred_rot | `(6, 64, 3, 3)` | 旋转矩阵 |
| 最终输出 | pred_xyz | `(1, 1, 6, 64, 3)` | (B, sets, samples, T, 3) |
| 最终输出 | pred_rot | `(1, 1, 6, 64, 3, 3)` | (B, sets, samples, T, 3, 3) |
| 文本输出 | extra["cot"] | `(1, 1, 6)` str array | CoC 推理文本 |

---

## 9. 文件索引

```
src/alpamayo1_5/
├── models/
│   ├── alpamayo1_5.py           # Alpamayo1_5 主模型类, Expert + 扩散推理
│   ├── base_model.py            # ReasoningVLA 基类, VLM 初始化, 轨迹融合
│   ├── action_in_proj.py        # PerWaypointActionInProjV2, 傅里叶编码+MLP
│   ├── token_utils.py           # Token 提取, 停止条件, padding 处理
│   └── delta_tokenizer.py       # DeltaTrajectoryTokenizer, delta 量化编码
├── diffusion/
│   ├── base.py                  # BaseDiffusion 抽象基类
│   └── flow_matching.py         # FlowMatching, Euler 积分采样
├── action_space/
│   ├── action_space.py          # ActionSpace 抽象基类
│   ├── unicycle_accel_curvature.py  # 单车运动学模型 (accel + curvature)
│   └── utils.py                 # Tikhonov 正则化求解工具
├── geometry/
│   └── rotation.py              # SO(3) 旋转工具: yaw 提取, 2D/3D 转换
├── config.py                    # Alpamayo1_5Config 配置类
├── helper.py                    # Message 构建, Processor 获取, 工具函数
├── load_physical_aiavdataset.py # 数据集加载
├── test_inference.py            # 端到端推理示例脚本
├── nav_utils.py                 # 导航文本处理 (CFG)
└── viz_utils.py                 # 可视化工具
```

---

## 10. Qwen3-VL 图像 vs 视频处理差异 & Alpamayo-1.5 的输入方式

### 10.1 结论先行：Alpamayo-1.5 把每一帧当作独立图像处理

Alpamayo-1.5 将 4 个摄像头 × 4 帧 = **16 张独立图像** 传给 Qwen3-VL Processor，**不使用视频通道**。

**证据** — `helper.py:71`（`_build_image_content()`）：

```python
# 逐帧构建 content, type 始终是 "image" 而非 "video"
content.append({"type": "text", "text": f"frame {frame_idx} "})
content.append({"type": "image", "image": frame})   # ← 每一帧都是独立图像
```

**调用路径** — `test_inference.py:36`：

```python
messages = helper.create_message(
    frames=data["image_frames"].flatten(0, 1),  # (N_cam, F, C, H, W) → (N_cam*F, C, H, W)
    camera_indices=data["camera_indices"]
)
```

`.flatten(0, 1)` 将 `(4_cam, 4_frame, C, H, W)` 展平为 `(16, C, H, W)`，然后逐张以 `{"type": "image"}` 传入 processor。**没有任何地方使用 `{"type": "video"}` 或 `pixel_values_videos`**。

### 10.2 Qwen3-VL 中图像与视频的处理差异

#### 10.2.1 Preprocessing 阶段

| 方面 | 图像 (`image_processor`) | 视频 (`video_processor`) |
|------|-------------------------|-------------------------|
| **输入** | 单帧 `(C, H, W)` | 多帧 `(T, C, H, W)` |
| **时间维度处理** | 单帧复制到 T=2 以满足 `temporal_patch_size=2` 的整除要求 | 多帧 padding 到 T 为 `temporal_patch_size=2` 的倍数 |
| **Conv3d 后的 `grid_t`** | **恒为 1** (`2 // 2 = 1`) | **≥ 1** (`num_frames // 2`) |
| **输出 `grid_thw`** | `[1, H', W']` | `[T', H', W']`，`T' = num_frames // 2` |
| **输出 key** | `pixel_values` + `image_grid_thw` | `pixel_values_videos` + `video_grid_thw` |

**代码位置**：
- 图像：`image_processing_qwen2_vl.py:273-297`
- 视频：`video_processing_qwen3_vl.py:234-262`

图像 preprocessing 的关键代码：

```python
# image_processing_qwen2_vl.py:273-280
patches = np.array(processed_images)  # shape: (1, C, H, W) — 单帧
if patches.shape[0] % temporal_patch_size != 0:
    # 单帧不能被 temporal_patch_size=2 整除 → 复制最后一帧
    repeats = np.repeat(patches[-1][np.newaxis], temporal_patch_size - 1, axis=0)
    patches = np.concatenate([patches, repeats], axis=0)  # → (2, C, H, W)
grid_t = patches.shape[0] // temporal_patch_size  # 2 // 2 = 1  ← 恒为 1
```

#### 10.2.2 ViT Encoder 阶段 — 完全相同

**关键发现**：`get_video_features()` 直接调用了 `get_image_features()`（`modular_qwen3_vl.py:1005-1006`）：

```python
def get_video_features(self, pixel_values_videos, video_grid_thw):
    # Same implementation as for images
    return self.get_image_features(pixel_values_videos, video_grid_thw)
```

ViT 编码器 (`Qwen3VLVisionModel`) 对图像和视频的处理完全一致：
- 都使用 `Conv3d(kernel=[2, 16, 16], stride=[2, 16, 16])` 做 patch embedding
- 都使用 2D 空间 RoPE (H, W)，**ViT 内部无时间维度编码**
- 差异仅在于 `grid_t`：图像恒为 1，视频可 >1

#### 10.2.3 Processor Token 展开 — 关键差异

**图像**（`processing_qwen3_vl.py:186-194`）：

```python
# 每张图像展开为 N 个 <|image_pad|>
num_image_tokens = image_grid_thw[index].prod() // merge_length
# grid_thw = [1, H', W'] → N = 1 * H' * W' / 4

# 模板中的展开结果:
# <|vision_start|><|image_pad|>×N<|vision_end|>
```

**视频**（`processing_qwen3_vl.py:217-231`）：

```python
# 每个视频按帧拆分, 每帧之间插入时间戳
for frame_idx in range(video_grid_thw[index][0]):
    curr_time = curr_timestamp[frame_idx]
    video_placeholder += f"<{curr_time:.1f} seconds>"           # ← 时间戳 token
    video_placeholder += vision_start_token + "<|placeholder|>" * frame_seqlen + vision_end_token

# 模板中的展开结果 (以 4 帧视频为例):
# <0.0 seconds><|vision_start|><|video_pad|>×M<|vision_end|>
# <0.5 seconds><|vision_start|><|video_pad|>×M<|vision_end|>
# <1.0 seconds><|vision_start|><|video_pad|>×M<|vision_end|>
# <1.5 seconds><|vision_start|><|video_pad|>×M<|vision_end|>
```

#### 10.2.4 LLM 3D RoPE Position IDs — 关键差异

**代码位置**：`modular_qwen3_vl.py:858-975`（`get_rope_index()`）

Qwen3-VL 使用 3 分量 MRoPE: `[Temporal, Height, Width]`，`mrope_section = [24, 20, 20]`。

**对于图像**（`grid_thw = [1, H', W']`）：

```
image token 的 3D position_ids:
  T维: [0, 0, 0, ..., 0]        ← 全部为 0, 因为 grid_t = 1
  H维: [0, 0, ..., 1, 1, ..., H'-1]   ← 空间行坐标
  W维: [0, 1, ..., W'-1, 0, 1, ...]   ← 空间列坐标
```

**对于视频**（经过 `get_rope_index` 中的拆分处理后，也是 `grid_thw = [1, H', W']` per frame）：

```python
# get_rope_index() 第 868-870 行:
video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
video_grid_thw[:, 0] = 1   # ← 将每个视频拆成多个 T=1 的帧
```

```
视频帧 0 的 3D position_ids:
  T维: [0, 0, ..., 0]        ← 全部为 0
  H维: [0, 0, ..., H'-1]
  W维: [0, 1, ..., W'-1]

(时间戳 text token: "<0.5 seconds>" 的 position 递增)

视频帧 1 的 3D position_ids:
  T维: [0, 0, ..., 0]        ← 仍然全部为 0 (时间靠 timestamp text token 编码)
  H维: [0, 0, ..., H'-1]
  W维: [0, 1, ..., W'-1]
```

**注释原文** (`modular_qwen3_vl.py:939`)：

> `t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)`

#### 10.2.5 总结对比

```
                     图像模式                              视频模式
                     ========                              ========

Processor 输出:      pixel_values                          pixel_values_videos
Grid:                image_grid_thw = [1, H', W']          video_grid_thw = [T', H', W']

ViT 编码:            完全相同 (Conv3d + spatial RoPE + Transformer Encoder)

LLM token 模板:      <|vision_start|>                      <0.0 seconds>
                     <|image_pad|> × N                     <|vision_start|>
                     <|vision_end|>                        <|video_pad|> × M
                                                          <|vision_end|>
                                                          <0.5 seconds>
                                                          <|vision_start|>
                                                          <|video_pad|> × M
                                                          <|vision_end|>
                                                          ...

LLM RoPE T维:       恒为 0                                恒为 0 (拆成了单帧)

帧间时间信息:        无                                     通过 timestamp text token
                                                          "<0.5 seconds>" 编码

LLM 中的 token:     <|image_pad|>                         <|video_pad|>
                    (token_id 不同!)                       (token_id 不同!)
```

### 10.3 Alpamayo-1.5 使用图像模式的具体影响

由于 Alpamayo-1.5 把所有帧当作独立图像：

**1. 每帧独立通过 ViT**

每张图片各自做 resize → 复制到 T=2 → Conv3d → ViT Encoder。不同帧之间**在 ViT 阶段没有信息交互**。

**2. 每帧在 LLM 中是独立的 vision block**

token 序列中每一帧的结构为：

```
"frame 0 " <|vision_start|><|image_pad|>×N<|vision_end|>
"frame 1 " <|vision_start|><|image_pad|>×N<|vision_end|>
...
```

帧之间的时序关系由**文本标注**隐式编码 — `"Front camera: "` + `"frame 0 "` / `"frame 1 "` / `"frame 2 "` / `"frame 3 "` 这些文本 token 告诉模型帧的顺序和摄像头归属。

**3. 没有时间戳 token**

与视频模式的 `<0.5 seconds>` 不同，图像模式不插入时间戳。帧间的时间关系完全靠 `"frame 0"` `"frame 1"` 等文本和 LLM 的因果位置递增来表达。

**4. LLM 3D RoPE 中所有帧的 T 维均为 0**

```
Camera 0, Frame 0:  T=[0,0,...,0]  H=[0,0,...,H'-1]  W=[0,1,...,W'-1]
Camera 0, Frame 1:  T=[0,0,...,0]  H=[0,0,...,H'-1]  W=[0,1,...,W'-1]  ← T维仍为0
Camera 0, Frame 2:  T=[0,0,...,0]  H=[0,0,...,H'-1]  W=[0,1,...,W'-1]  ← T维仍为0
Camera 0, Frame 3:  T=[0,0,...,0]  H=[0,0,...,H'-1]  W=[0,1,...,W'-1]  ← T维仍为0
```

不同帧的 image token 在 position space 中的区分靠**帧间的文本 token 产生的 position 偏移**，而不是 T 维坐标。

**5. 为什么选择图像模式而非视频模式？**

- **训练一致性**：模型训练时就是以独立图像方式处理的，推理时必须保持一致
- **灵活性**：多摄像头帧来自不同视角（FOV 不同），不适合作为同一个视频的连续帧
- **分辨率独立**：每张图片可以有不同分辨率，图像模式允许这一点；视频模式要求所有帧同尺寸

---

## 11. 进入 VLM Backbone 的完整 Token 序列结构

> 本节是第 10 节中 token 布局的完整展开版本。

经过 Qwen3-VL Processor tokenize + `fuse_traj_tokens()` 轨迹 token 融合后，`input_ids` 的完整结构如下（Qwen3-VL chat template 格式）：

```
┌─ System Message ──────────────────────────────────────────────────────────┐
│ <|im_start|>system\n                                                      │
│ You are a driving assistant that generates safe and accurate actions.      │
│ <|im_end|>\n                                                              │
├─ User Message ────────────────────────────────────────────────────────────┤
│ <|im_start|>user\n                                                        │
│                                                                           │
│ ── Camera 0 (Front left camera) ──                                        │
│ "Front left camera: "                                                     │
│ "frame 0 " <|vision_start|><|image_pad|>×N₀<|vision_end|>                │
│ "frame 1 " <|vision_start|><|image_pad|>×N₁<|vision_end|>                │
│ "frame 2 " <|vision_start|><|image_pad|>×N₂<|vision_end|>                │
│ "frame 3 " <|vision_start|><|image_pad|>×N₃<|vision_end|>                │
│                                                                           │
│ ── Camera 1 (Front camera) ──                                             │
│ "Front camera: "                                                          │
│ "frame 0 " <|vision_start|><|image_pad|>×M₀<|vision_end|>                │
│ "frame 1 " ...                                                            │
│ ...                                                                       │
│                                                                           │
│ ── Camera 2, 3 同理 (共 N_cam 个摄像头, 每个 4 帧) ──                    │
│ ...                                                                       │
│                                                                           │
│ ── 轨迹历史 (48 个离散 token) ──                                          │
│ <|traj_history_start|>                                                    │
│ <i23><i156><i789><i45>...(共 48 个, 16步 × 3维xyz)                        │
│ <|traj_history_end|>                                                      │
│                                                                           │
│ ── 导航指令 (可选) ──                                                     │
│ <|route_start|>Turn left onto De La Cruz Boulevard<|route_end|>           │
│                                                                           │
│ ── Prompt 指令 ──                                                         │
│ output the chain-of-thought reasoning of the driving process,             │
│ then output the future trajectory.                                        │
│ <|im_end|>\n                                                              │
├─ Assistant Prefix (强制模型从 CoC 推理开始生成) ──────────────────────────┤
│ <|im_start|>assistant\n                                                   │
│ <|cot_start|>                                                             │
└───────────────────────────────────────────────────────────────────────────┘
```

**关键说明**：

1. **图像 token 数量**：每张图像的 `<|image_pad|>` 数量由 Qwen3-VL Processor 根据图片分辨率自动计算，受 `min_pixels=163840` ~ `max_pixels=196608` 约束。图像在 VLM 内部会被 ViT 编码成对应数量的 vision embedding，再通过 `masked_scatter` 替换到 `<|image_pad|>` 的位置上。

2. **轨迹 token**：48 个 `<i{n}>` token 来自 `DeltaTrajectoryTokenizer`，编码流程为：相邻时间步做 delta → 归一化到 `[0, 1]` → 量化到 `[0, 999]` → 展平 `(16步 × 3维)` = 48 个 token → 加上 `traj_token_start_idx` 偏移映射到词汇表 ID。

3. **导航部分**：`<|route_start|>...<|route_end|>` 仅在提供 `nav_text` 时存在；CFG 推理时会构建一份去掉导航的序列用于 unguided 分支。

4. **Assistant prefix**：`<|cot_start|>` 强制模型从 Chain-of-Causation 推理开始生成，VLM 随后自回归生成推理文本直到输出 `<|traj_future_start|>`。

**VLM 自回归生成后的完整序列**：

```
[上述所有输入 token][cot_start] The vehicle ahead is slowing down. The ego
vehicle should maintain safe distance and prepare to brake. [cot_end]
[traj_future_start] <extra_token>
                     ↑ StopAfterEOS 多生成的 1 个 token (为了 KV Cache 完整)
```

---

## 12. 全流程 Attention Mask 详解（VLM + Expert）

Alpamayo-1.5 有两个注意力阶段，mask 策略完全不同：

### 12.1 阶段一：VLM 自回归生成 — 标准因果注意力 (Causal Attention)

VLM（Qwen3-VL）在 `generate()` 过程中使用**标准的因果掩码**，即每个 token 只能 attend to 自己和前面的所有 token。

**掩码示意** (✓ = 可见, ✗ = 不可见)：

```
              sys₀ sys₁ usr₀ img₀ img₁ ... hist₀ ... cot_start CoC₁ CoC₂ ... traj_fut_start
sys₀           ✓    ✗    ✗    ✗    ✗       ✗          ✗         ✗    ✗         ✗
sys₁           ✓    ✓    ✗    ✗    ✗       ✗          ✗         ✗    ✗         ✗
usr₀           ✓    ✓    ✓    ✗    ✗       ✗          ✗         ✗    ✗         ✗
img₀           ✓    ✓    ✓    ✓    ✗       ✗          ✗         ✗    ✗         ✗
img₁           ✓    ✓    ✓    ✓    ✓       ✗          ✗         ✗    ✗         ✗
...
hist₀          ✓    ✓    ✓    ✓    ✓  ...  ✓          ✗         ✗    ✗         ✗
...
cot_start      ✓    ✓    ✓    ✓    ✓  ...  ✓     ...  ✓         ✗    ✗         ✗
CoC₁           ✓    ✓    ✓    ✓    ✓  ...  ✓     ...  ✓         ✓    ✗         ✗
CoC₂           ✓    ✓    ✓    ✓    ✓  ...  ✓     ...  ✓         ✓    ✓         ✗
...
traj_fut_start ✓    ✓    ✓    ✓    ✓  ...  ✓     ...  ✓         ✓    ✓    ...  ✓
```

**补充说明**：

- **同一张图片内部的 image token 之间也是因果的**：`img_token_k` 只能看到 `img_token_0` ~ `img_token_k`，看不到后续的 `img_token_{k+1}`。图像 patch 之间的双向信息交互已经在**独立的 ViT Encoder 阶段**完成（ViT 内部使用标准 Transformer Encoder 的 bidirectional self-attention），编码后的 feature 再被注入到 LLM token 位置上，此后遵循 LLM 的因果规则。
- **轨迹历史 token** (`<i23>`, `<i156>` 等) 也遵循因果规则，它们之间不存在特殊的双向注意力。
- **Left-padding 位置**的 mask 值为 0 (不可见)，由 1D `attention_mask` 中的 0 值转化而来。

### 12.2 阶段二：Expert Transformer — 非因果注意力 + Cross-Attend to VLM KV Cache

Expert 处理 64 个扩散 token，通过 `past_key_values` 复用 VLM 的 KV Cache。

**4D attention mask shape**: `(B*ns, 1, 64, KV_len + 64)`

**掩码布局**：

```
                ←───────── VLM KV Cache (KV_len) ──────────→  ←── 64个扩散token ──→

KV 内容:  [pad..] [sys] [img...] [traj_hist] [prompt] [CoC...] [cot_end] [traj_fut_start] [pad..gap..] │ [diff₀] [diff₁] ... [diff₆₃]
           ↑                                                                             ↑              ↑
        可能的                                                                        offset          KV_len
        left-pad

每个扩散 token 的注意力 (✓ = 可见, ✗ = masked 为 -inf):

              [pad] [sys] [img...] [traj_hist] [CoC...] [cot_end] [traj_fut_start] [pad..gap..] [diff₀] [diff₁] ... [diff₆₃]
diff₀          ✗     ✓      ✓         ✓          ✓         ✓            ✗              ✗          ✓       ✓           ✓
diff₁          ✗     ✓      ✓         ✓          ✓         ✓            ✗              ✗          ✓       ✓           ✓
diff₂          ✗     ✓      ✓         ✓          ✓         ✓            ✗              ✗          ✓       ✓           ✓
...            ...
diff₆₃         ✗     ✓      ✓         ✓          ✓         ✓            ✗              ✗          ✓       ✓           ✓
```

**关键特性**：

1. **所有 64 个扩散 token 的注意力模式完全相同** — 非因果 (`is_causal=False`)，64 个 token 之间双向可见。
2. **每个扩散 token 可以看到**：
   - VLM KV Cache 中位置 `0` 到 `offset-1` 的全部有效 token（系统文本、所有图像 token、轨迹历史 token、prompt 文本、CoC 推理文本、`<|cot_end|>`）
   - 所有 64 个扩散 token（互相可见，双向注意力）
3. **每个扩散 token 看不到**：
   - `offset` 到 `KV_len-1` 之间的 gap（`<|traj_future_start|>` 及其后的 padding）
   - Left-padding 位置
4. **KV Cache 管理**：每个扩散步结束后调用 `prompt_cache.crop(prefill_seq_len)` 裁剪掉新增的 KV，恢复到仅含 VLM 前缀的状态，使得 10 步扩散循环都复用同一份 KV Cache。

**为什么 offset 在 batch 内不同？** 因为每个样本的 VLM 生成长度不等（CoC 推理文本长短各异），`<|traj_future_start|>` 出现在不同位置，但 KV Cache 是按最长序列 padding 的，所以中间会产生长度不一的 gap。

### 12.3 两阶段注意力对比总结

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        VLM 阶段 (因果注意力)                                │
│                                                                              │
│  掩码规则:  kv_idx <= q_idx  (标准下三角因果掩码)                            │
│  原因:     VLM 需要自回归逐 token 生成 CoC 推理文本                          │
│                                                                              │
│  [sys][img×N][traj_hist×48][nav][prompt]→[cot_start]→[CoC 文本]→             │
│  →[cot_end]→[traj_future_start]                                             │
│                                                                              │
│  所有计算结果累积在 KV Cache 中                                              │
└────────────────────────────────────┬─────────────────────────────────────────┘
                                     │ KV Cache (只读复用)
                                     ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                      Expert 阶段 (非因果注意力)                              │
│                                                                              │
│  掩码规则:  自定义 4D mask, 扩散 token 间完全双向可见                        │
│  原因:     64 个路点同时输入同时输出, 需要双向协调以保证轨迹时序一致          │
│                                                                              │
│  64 个扩散 token:                                                            │
│    ✓ 互相完全可见 (双向)                                                     │
│    ✓ 能看到 VLM 的全部有效上下文 (图像 + 历史 + CoC 推理 + 文本)             │
│    ✗ 看不到 <|traj_future_start|> 及其后的 padding                           │
│    ✗ 看不到 left-padding                                                     │
│                                                                              │
│  每个扩散步都复用同一份 VLM KV Cache (每步结束后 crop 掉新增的 KV)           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. VLM Causal Mask 代码溯源与设计原因

### 13.1 调用链

从 Alpamayo 到最终生成 4D 因果掩码的完整调用链：

```
Alpamayo1_5.sample_trajectories_from_data_with_vlm_rollout()
│   alpamayo1_5.py:285
│   self.vlm.generate(input_ids, attention_mask=1D_padding_mask, ...)
│
└─→ Qwen3VLForConditionalGeneration.generate()
    │   HuggingFace GenerationMixin
    │
    └─→ Qwen3VLModel.forward()
        │   modular_qwen3_vl.py:1010
        │   图像 embedding 通过 masked_scatter 替换 <|image_pad|> 位置
        │
        └─→ Qwen3VLTextLanguageModel.forward()
            │   modular_qwen3_vl.py:804
            │
            └─→ create_causal_mask(
                    config=self.config,
                    input_embeds=inputs_embeds,
                    attention_mask=attention_mask,  ← 1D padding mask (B, L)
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    position_ids=text_position_ids,
                    # 注意: or_mask_function 和 and_mask_function 均未传入, 默认 None
                )
                │   masking_utils.py:745
                │
                └─→ mask_interface(mask_function=causal_mask_function, ...)
                    │   根据 attn_implementation 分发 (eager / sdpa / flash_attention_2)
                    │
                    └─→ 最终输出 4D mask: (B, 1, Q_len, KV_len)
```

### 13.2 核心代码位置

**文件**: `transformers/masking_utils.py`

**因果掩码工厂函数** (第 74-78 行):

```python
def causal_mask_function(batch_idx, head_idx, q_idx, kv_idx):
    """标准下三角因果掩码"""
    return kv_idx <= q_idx
```

**`create_causal_mask()`** (第 745-836 行):

```python
def create_causal_mask(config, input_embeds, attention_mask, cache_position,
                       past_key_values, position_ids=None,
                       or_mask_function=None, and_mask_function=None):

    mask_factory_function = causal_mask_function    # ← 基础: kv_idx <= q_idx

    # Qwen3-VL 未传入 or_mask_function → 不会执行
    if or_mask_function is not None:
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)

    # Qwen3-VL 未传入 and_mask_function → 不会执行
    if and_mask_function is not None:
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)

    # 生成 4D mask
    causal_mask = mask_interface(
        mask_function=mask_factory_function,  # 纯因果, 无任何覆盖
        attention_mask=attention_mask,         # 1D padding mask → 同时屏蔽 padding 位置
        ...
    )
    return causal_mask  # shape: (B, 1, Q_len, KV_len)
```

**文件**: `transformers/models/qwen3_vl/modular_qwen3_vl.py`

**调用点** (第 804-811 行):

```python
# Qwen3VLTextLanguageModel.forward() 中
attention_mask = create_causal_mask(
    config=self.config,
    input_embeds=inputs_embeds,
    attention_mask=attention_mask,   # 1D (B, L), 值 0/1
    cache_position=cache_position,
    past_key_values=past_key_values,
    position_ids=text_position_ids,
    # or_mask_function=None   ← 未传, 图像 token 无特殊双向处理
    # and_mask_function=None  ← 未传
)
```

### 13.3 为什么 VLM 使用 Causal Mask 而非 Bidirectional

**原因一：自回归生成的本质要求**

VLM 的核心任务是**逐 token 自回归生成 CoC 推理文本**。自回归生成要求模型在预测第 `t` 个 token 时，只能看到 `0` ~ `t-1` 的上下文。如果使用 bidirectional mask，模型在预测时就能"偷看"未来 token，导致训练和推理行为不一致。

**原因二：Decoder-Only 架构的训练方式决定**

Qwen3-VL 的 backbone 是 Qwen3（decoder-only LLM），**从预训练到微调始终使用 causal mask**。如果推理时更换为 bidirectional mask，模型会遇到训练时从未见过的注意力模式，导致输出质量严重下降。

**原因三：图像 token 无需在 LLM 层做双向注意力**

从代码确认，Qwen3-VL **对图像 token 没有任何特殊的 mask 处理**：
- 未传入 `or_mask_function`（用于在因果掩码上叠加双向区域）
- 未传入 `and_mask_function`
- 图像处理方式是 **embedding 替换** (`masked_scatter`)，不修改 attention mask

这意味着同一张图片内部的 image token 之间的注意力也是因果的。图像 patch 间的双向信息交互已经在**独立的 ViT Encoder 阶段** (`Qwen3VLVisionModel`) 完成 — ViT 内部使用标准 Transformer Encoder 的 bidirectional self-attention。编码后的 feature 注入 LLM token 位置后，就遵循 LLM 的因果规则。

**对比：Expert 为什么可以用非因果？**

Expert 不做自回归文本生成。它的 64 个扩散 token 是**同时输入、同时输出**的（类似 Encoder 的使用方式）。每个路点需要感知所有其他路点的信息才能生成时序一致的轨迹，因此使用 `is_causal=False` 的双向注意力是合理的。

### 13.4 1D Padding Mask 到 4D Causal Mask 的转换过程

Alpamayo 传入 VLM 的 `attention_mask` 是 1D 的 `(B, L)`，值为 `0`（padding）/ `1`（有效）。转换为 4D mask 的过程：

```
输入: attention_mask = [0, 0, 1, 1, 1, ..., 1]    ← (B, L), left-padding 示例
                        ↑  ↑
                       padding

      ┌─────────────────────────────────────────────┐
      │ _preprocess_mask_arguments()                 │
      │   masking_utils.py:664                       │
      │   - 检查是否已是 4D (是则直接返回)            │
      │   - 转为 bool dtype                          │
      │   - 计算 kv_length, kv_offset                │
      └──────────────────┬──────────────────────────┘
                         │
                         ▼
      ┌─────────────────────────────────────────────┐
      │ mask_interface() 分发到具体后端               │
      │   - eager: 直接构造 4D tensor                │
      │   - sdpa: sdpa_mask_recent_torch()           │
      │   - flash_attention_2: 特殊处理              │
      │                                              │
      │ 核心逻辑:                                     │
      │   mask[b, h, q, kv] =                        │
      │     causal_mask_function(b, h, q, kv)        │
      │     AND padding_mask[b, kv] == 1             │
      │                                              │
      │   即: (kv_idx <= q_idx) AND (非padding位置)   │
      └──────────────────┬──────────────────────────┘
                         │
                         ▼
      输出: 4D mask (B, 1, Q_len, KV_len)
            - 0.0   = 可以 attend
            - -inf  = 不可 attend (padding 或 future token)
```
