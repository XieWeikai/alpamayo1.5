# Qwen3-VL 核心技术详解：MRoPE 与 DeepStack

本文档以 Qwen3-VL-8B-Instruct 为例，详细解析 Qwen3-VL 的两项核心技术：**Multimodal Rotary Position Embedding (MRoPE)** 和 **DeepStack Visual Feature Injection**。所有 tensor shape 均标注为具体数值（以 8B 模型默认配置为基准）。

> **代码来源**：HuggingFace Transformers 中 `transformers/models/qwen3_vl/modular_qwen3_vl.py` 为源文件，`modeling_qwen3_vl.py` 为 CI 自动展开的等价文件。本文档引用的行号均来自 `modular_qwen3_vl.py`。

---

## 目录

1. [模型配置参数速查](#1-模型配置参数速查)
2. [MRoPE：多模态旋转位置编码](#2-mrope多模态旋转位置编码)
   - 2.1 [与标准 1D RoPE 的区别](#21-与标准-1d-rope-的区别)
   - 2.2 [3D Position IDs 的构建 (get_rope_index)](#22-3d-position-ids-的构建)
   - 2.3 [频率计算与 Interleaving](#23-频率计算与-interleaving)
   - 2.4 [在 Attention 中的应用](#24-在-attention-中的应用)
   - 2.5 [rope_deltas 的含义](#25-rope_deltas-的含义)
   - 2.6 [完整数据流图](#26-完整数据流图)
3. [DeepStack：视觉特征深度注入](#3-deepstack视觉特征深度注入)
   - 3.1 [设计思想](#31-设计思想)
   - 3.2 [ViT 中间层特征提取](#32-vit-中间层特征提取)
   - 3.3 [PatchMerger 投影](#33-patchmerger-投影)
   - 3.4 [LLM Decoder 中的注入](#34-llm-decoder-中的注入)
   - 3.5 [完整数据流图](#35-完整数据流图)
4. [两项技术的关联](#4-两项技术的关联)

---

## 1. 模型配置参数速查

### Qwen3VLTextConfig 默认值 (Line 208-251)

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `hidden_size` | 4096 | LLM 隐藏维度 |
| `num_hidden_layers` | 32 | LLM decoder 层数 |
| `num_attention_heads` | 32 | 注意力头数 |
| `num_key_value_heads` | 32 | KV 头数 (GQA) |
| `head_dim` | 128 | 每个头的维度 |
| `rope_theta` | 5,000,000.0 | RoPE 基础频率 |
| `mrope_section` | `[24, 20, 20]` | MRoPE 3D 频率分配 |

### Qwen3VLVisionConfig 默认值 (Line 66-101)

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `depth` | 27 | ViT 层数 |
| `hidden_size` | 1152 | ViT 隐藏维度 |
| `num_heads` | 16 | ViT 注意力头数 |
| `patch_size` | 16 | 空间 patch 大小 |
| `temporal_patch_size` | 2 | 时间 patch 大小 |
| `spatial_merge_size` | 2 | 空间合并因子 (2×2→1) |
| `out_hidden_size` | 3584* | ViT 输出维度 → LLM |
| `deepstack_visual_indexes` | `[8, 16, 24]` | DeepStack 提取层 |

> *注：默认值 3584 对应 4B 模型配置。对于 8B 模型，`out_hidden_size` 会被设为与 LLM `hidden_size` (4096) 匹配，因为 DeepStack 注入要求维度一致。

---

## 2. MRoPE：多模态旋转位置编码

### 2.1 与标准 1D RoPE 的区别

标准 RoPE 为每个 token 分配一个标量位置 `p`，然后用旋转矩阵编码到 Q/K 中。**MRoPE** 为每个 token 分配一个 **3D 位置向量 `(t, h, w)`**，分别表示时间、高度、宽度：

| 方面 | 标准 1D RoPE | 3D MRoPE |
|------|-------------|----------|
| Position ID shape | `(B, L)` | `(3, B, L)` — `[T, H, W]` |
| 频率向量 | `(head_dim//2,)` = `(64,)` | 同一组 `(64,)` 频率分别与 T/H/W 各做一次外积 |
| 频率布局 | 单一序列 | **Interleaved**：T/H/W 频率交织排列 |
| 最终 cos/sin shape | `(B, L, 128)` | `(B, L, 128)` — 但内部编码了 3D 信息 |

**核心思想**：

- 文本 token 的 T/H/W 三个维度共享同一个递增位置 → 退化为 1D
- 图像 token 的 T 恒为 0，H/W 按 2D 网格坐标赋值 → 编码空间位置
- 视频 token 类似图像（因为 Qwen3-VL 用 timestamp 文本编码时间）

### 2.2 3D Position IDs 的构建

**代码位置**：`get_rope_index()`，Line 858-975

#### 文本 token 的 Position IDs

```python
# Line 937
llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
```

三个维度完全相同：

```
text "The car is":
  T: [0, 1, 2, 3]
  H: [0, 1, 2, 3]    ← 三维相同
  W: [0, 1, 2, 3]
```

#### 图像 token 的 Position IDs

```python
# Line 940-943
t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
```

对于一张图片 `grid_thw = [1, 3, 4]`（T=1, H=3, W=4，共 12 个 vision token）：

```
T: [0, 0, 0, 0,  0, 0, 0, 0,  0, 0, 0, 0]    ← 图像 T 恒为 0
H: [0, 0, 0, 0,  1, 1, 1, 1,  2, 2, 2, 2]    ← 行坐标
W: [0, 1, 2, 3,  0, 1, 2, 3,  0, 1, 2, 3]    ← 列坐标
```

加上 `text_len + st_idx` 偏移后，三维各自加上相同的基准值。

#### 完整序列的 Position IDs 拼接示例

假设序列为：`[text×4] [image 3×4=12 tokens] [text×3]`

```
Token:     t₀  t₁  t₂  t₃  img₀ img₁ ... img₁₁  t₄  t₅  t₆
索引:       0   1   2   3   4    5        15     16  17  18

T维:        0   1   2   3   4   4  ...    4      16  17  18
H维:        0   1   2   3   4   4  ...    6      16  17  18
W维:        0   1   2   3   4   5  ...    7      16  17  18
                              ↑                    ↑
                    图像区域：T/H/W 各不同          后续文本：三维重新同步
```

**关键细节**：
- `st_idx = llm_pos_ids_list[-1].max() + 1`（Line 936）—— 每段的起始位置从上一段最大值 +1 开始
- 图像区域中 H 维的最大值是 `text_len + st_idx + llm_grid_h - 1`，W 维的最大值是 `text_len + st_idx + llm_grid_w - 1`，而 T 维始终是 `text_len + st_idx + 0`
- 后续文本段的三维重新同步到同一起点（`st_idx = max(T, H, W) + 1`）

#### rope_deltas 的计算

```python
# Line 953
mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
```

由于图像区域的 2D 网格使得某些维度的位置值较大（比序列长度 L 大），所以 `max_position > L`。`rope_deltas = max_position + 1 - L`，记录这个偏移量，在自回归生成阶段用于正确计算新 token 的位置。

### 2.3 频率计算与 Interleaving

**代码位置**：`Qwen3VLTextRotaryEmbedding.forward()`，Line 426-444

#### Step 1：将 inv_freq 扩展为 3D

```python
# inv_freq: (head_dim//2,) = (64,)

inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, B, -1, 1)
# (64,) → (1,1,64,1) → (3, B, 64, 1)

position_ids_expanded = position_ids[:, :, None, :].float()
# (3, B, L) → (3, B, 1, L)
```

#### Step 2：矩阵乘法计算频率

```python
freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
# (3, B, 64, 1) @ (3, B, 1, L) = (3, B, 64, L) → transpose → (3, B, L, 64)
```

此时 `freqs[0]`、`freqs[1]`、`freqs[2]` 分别包含 T/H/W 三个维度的频率，每个都是 `(B, L, 64)` 且使用相同的 `inv_freq` 基底。

#### Step 3：Interleaved MRoPE — 核心操作

**代码位置**：`apply_interleaved_mrope()`，Line 409-424

```python
def apply_interleaved_mrope(self, freqs, mrope_section):
    # freqs: (3, B, L, 64),  mrope_section: [24, 20, 20]
    freqs_t = freqs[0]  # 以 T 维频率为基底, shape: (B, L, 64)
    for dim, offset in enumerate((1, 2), start=1):  # dim=1(H), dim=2(W)
        length = mrope_section[dim] * 3              # 20*3 = 60
        idx = slice(offset, length, 3)               # H: [1,4,7,...,58]  W: [2,5,8,...,59]
        freqs_t[..., idx] = freqs[dim, ..., idx]     # 从 H/W 维频率中取对应位置覆盖
    return freqs_t  # (B, L, 64)
```

**逐步展示 64 维频率的变化**：

```
初始 freqs_t = freqs[0] (全为 T 维频率):
idx: [0   1   2   3   4   5   6   7   8  ...  58  59  60  61  62  63]
val: [T₀  T₁  T₂  T₃  T₄  T₅  T₆  T₇  T₈ ... T₅₈ T₅₉ T₆₀ T₆₁ T₆₂ T₆₃]

第一次循环 (dim=1, H): idx = slice(1, 60, 3) → 位置 [1, 4, 7, 10, ..., 58]
覆盖后:
idx: [0   1   2   3   4   5   6   7   8  ...  58  59  60  61  62  63]
val: [T₀  H₁  T₂  T₃  H₄  T₅  T₆  H₇  T₈ ... H₅₈ T₅₉ T₆₀ T₆₁ T₆₂ T₆₃]

第二次循环 (dim=2, W): idx = slice(2, 60, 3) → 位置 [2, 5, 8, 11, ..., 59]
覆盖后:
idx: [0   1   2   3   4   5   6   7   8  ...  58  59  60  61  62  63]
val: [T₀  H₁  W₂  T₃  H₄  W₅  T₆  H₇  W₈ ... H₅₈ W₅₉ T₆₀ T₆₁ T₆₂ T₆₃]
```

**最终 Interleaved 布局** (64 维)：

```
位置 0-59:  [T₀, H₁, W₂,  T₃, H₄, W₅,  T₆, H₇, W₈, ...]  ← 20 组 (T,H,W) 三元组
位置 60-63: [T₆₀, T₆₁, T₆₂, T₆₃]                            ← 剩余 4 个纯 T 维频率
```

分配：T 占 24 维 (20 个在三元组中 + 4 个尾部)，H 占 20 维，W 占 20 维，共 64 维。

> **注意**：这里 `freqs[dim, ..., idx]` 取的是**同一组 inv_freq 与不同 position_ids 计算出的频率值**，也就是说 inv_freq 基底相同但 position 不同（T 维的位置和 H 维的位置不同），所以频率值不同。

#### Step 4：拼接并计算 cos/sin

```python
emb = torch.cat((freqs, freqs), dim=-1)   # (B, L, 64) → (B, L, 128)
cos = emb.cos() * self.attention_scaling   # (B, L, 128)
sin = emb.sin() * self.attention_scaling   # (B, L, 128)
```

重复一次是 RoPE 的标准做法：`rotate_half` 操作需要前半和后半互相配对。

### 2.4 在 Attention 中的应用

**代码位置**：`Qwen3VLTextAttention.forward()`，Line 447-493

```python
# Q/K 投影
query_states = self.q_norm(self.q_proj(hidden_states).view(B, L, n_heads, head_dim)).transpose(1, 2)
# (B, L, 4096) → (B, L, 32, 128) → (B, 32, L, 128)

key_states = self.k_norm(self.k_proj(hidden_states).view(B, L, n_kv_heads, head_dim)).transpose(1, 2)
# (B, L, 4096) → (B, L, 32, 128) → (B, 32, L, 128)

# 应用 RoPE
cos, sin = position_embeddings   # 各 (B, L, 128)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

`apply_rotary_pos_emb` (继承自 `modeling_qwen3.py`)：

```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)       # (B, L, 128) → (B, 1, L, 128)
    sin = sin.unsqueeze(unsqueeze_dim)       # (B, L, 128) → (B, 1, L, 128)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
    # 输出：(B, 32, L, 128)
```

`rotate_half` 将 128 维分成前 64 和后 64，做 `[-x2, x1]` 交换。

**Tensor Shape 追踪**：

| 步骤 | Tensor | Shape |
|------|--------|-------|
| 输入 `hidden_states` | LLM 隐藏状态 | `(B, L, 4096)` |
| Q projection | `query_states` | `(B, 32, L, 128)` |
| K projection | `key_states` | `(B, 32, L, 128)` |
| RoPE cos/sin | `position_embeddings` | `(B, L, 128)` each |
| cos/sin unsqueeze | 广播到 heads | `(B, 1, L, 128)` |
| `q * cos` | 逐元素乘 | `(B, 32, L, 128)` |
| `rotate_half(q) * sin` | 旋转半部分 | `(B, 32, L, 128)` |
| 输出 Q/K | 旋转后 | `(B, 32, L, 128)` |

### 2.5 rope_deltas 的含义

`rope_deltas` shape: `(B, 1)`

**作用**：在 VLM 自回归 `generate()` 的 decode 阶段，每步只处理一个新 token。此时需要为新 token 计算正确的 position_id，但由于 prefill 阶段图像 token 的 2D 网格占用了额外的位置空间，decode 的位置不是简单的 `len(input_ids)`，而是 `len(input_ids) + rope_deltas`。

```python
# modular_qwen3_vl.py Line 1114 (Qwen3VLModel.forward, decode 阶段)
delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
position_ids = delta.view(1, -1, 1).expand(3, -1, -1)
```

### 2.6 完整数据流图

```
                           Position IDs 构建
                           ================

 input_ids: (B, L)
 image_grid_thw: (N_img, 3)
         │
         ▼
  get_rope_index()  (Line 858-975)
         │
         ├─ 文本段: position_ids = arange(text_len).expand(3, -1)  ← T=H=W 相同
         ├─ 图像段: t_index=[0,0,...], h_index=[行坐标], w_index=[列坐标]
         ├─ 拼接: position_ids shape = (3, B, L)
         └─ rope_deltas = max_pos + 1 - L,  shape = (B, 1)
         │
         ▼
                        频率计算与 Interleaving
                        ======================

  Qwen3VLTextRotaryEmbedding.forward()  (Line 426-444)
         │
         ├─ inv_freq_expanded: (64,) → (3, B, 64, 1)
         ├─ position_ids_expanded: (3, B, L) → (3, B, 1, L)
         ├─ freqs = matmul: (3, B, 64, 1) @ (3, B, 1, L) → (3, B, L, 64)
         │
         ├─ apply_interleaved_mrope():
         │    freqs_t = freqs[0]                              (B, L, 64)
         │    freqs_t[..., 1::3 up to 60] = freqs[1, ..., ]  ← 插入 H 频率
         │    freqs_t[..., 2::3 up to 60] = freqs[2, ..., ]  ← 插入 W 频率
         │    → interleaved: (B, L, 64)
         │
         ├─ emb = cat(freqs, freqs): (B, L, 128)
         ├─ cos = emb.cos(): (B, L, 128)
         └─ sin = emb.sin(): (B, L, 128)
         │
         ▼
                         Attention 中的应用
                         =================

  Qwen3VLTextAttention.forward()  (Line 447-493)
         │
         ├─ Q: (B, L, 4096) → q_proj → q_norm → (B, 32, L, 128)
         ├─ K: (B, L, 4096) → k_proj → k_norm → (B, 32, L, 128)
         │
         └─ apply_rotary_pos_emb(Q, K, cos, sin):
              cos: (B, L, 128) → unsqueeze → (B, 1, L, 128)
              sin: (B, L, 128) → unsqueeze → (B, 1, L, 128)
              Q_rot = Q * cos + rotate_half(Q) * sin  → (B, 32, L, 128)
              K_rot = K * cos + rotate_half(K) * sin  → (B, 32, L, 128)
```

---

## 3. DeepStack：视觉特征深度注入

### 3.1 设计思想

标准 VLM 的做法是：ViT 编码图像 → 最终输出的 vision embedding 替换 LLM 中对应位置的 token embedding → LLM 从第 0 层开始处理。**问题**：ViT 的中间层特征（低层纹理、中层结构）在这个过程中被丢弃了。

**DeepStack** 的解决方案：从 ViT 的**多个中间层**提取特征，投影到 LLM 维度后，在 LLM 的**前几层 decoder 之后**残差注入到 vision token 的 hidden states 中。

```
ViT Layer 8 的特征  → merger[0] 投影 → 注入到 LLM Layer 0 之后
ViT Layer 16 的特征 → merger[1] 投影 → 注入到 LLM Layer 1 之后
ViT Layer 24 的特征 → merger[2] 投影 → 注入到 LLM Layer 2 之后
```

### 3.2 ViT 中间层特征提取

**代码位置**：`Qwen3VLVisionModel.forward()`，Line 676-726

```python
def forward(self, hidden_states, grid_thw, **kwargs):
    hidden_states = self.patch_embed(hidden_states)       # → (seq_len, 1152)
    hidden_states = hidden_states + pos_embeds             # + 位置嵌入

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):          # 遍历 27 层 ViT
        hidden_states = blk(hidden_states, ...)            # → (seq_len, 1152)

        if layer_num in self.deepstack_visual_indexes:     # [8, 16, 24]
            deepstack_feature = self.deepstack_merger_list[
                self.deepstack_visual_indexes.index(layer_num)
            ](hidden_states)                                # → (seq_len', out_hidden_size)
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = self.merger(hidden_states)             # 最终输出 → (seq_len', out_hidden_size)
    return hidden_states, deepstack_feature_lists
```

**Tensor Shape 追踪**（以单张 H'=14, W'=14 的图片为例，`grid_thw=[1,14,14]`）：

| 步骤 | Shape | 说明 |
|------|-------|------|
| `pixel_values` 输入 | `(196, 3×2×16×16)` = `(196, 1536)` | 196 = 14×14 个 patch |
| `patch_embed` 后 | `(196, 1152)` | Conv3d 投影 |
| `+ pos_embeds` 后 | `(196, 1152)` | 加位置嵌入 |
| ViT Block 0-7 | `(196, 1152)` | 不变 |
| **Layer 8 → merger[0]** | `(196, 1152)` → `(49, out_hidden_size)` | spatial merge 2×2 → 196/4=49 |
| ViT Block 9-15 | `(196, 1152)` | 不变 |
| **Layer 16 → merger[1]** | `(196, 1152)` → `(49, out_hidden_size)` | 同上 |
| ViT Block 17-23 | `(196, 1152)` | 不变 |
| **Layer 24 → merger[2]** | `(196, 1152)` → `(49, out_hidden_size)` | 同上 |
| ViT Block 25-26 | `(196, 1152)` | 不变 |
| 最终 `merger` | `(196, 1152)` → `(49, out_hidden_size)` | 主输出 |

### 3.3 PatchMerger 投影

**代码位置**：`Qwen3VLVisionPatchMerger`，Line 357-370

每个 PatchMerger 做两件事：**(1) 2×2 空间合并** + **(2) MLP 投影到 LLM 维度**。

```python
class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm=False):
        self.hidden_size = config.hidden_size * (config.spatial_merge_size ** 2)
        # = 1152 * 4 = 4608

        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size
        )
        self.linear_fc1 = nn.Linear(4608, 4608)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(4608, out_hidden_size)

    def forward(self, x):
        # x: (seq_len, 1152)
        x = self.norm(
            x.view(-1, 4608) if self.use_postshuffle_norm else x
        ).view(-1, 4608)
        # → (seq_len/4, 4608)  — 2×2 patch 被拼接成一个
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        # → (seq_len/4, out_hidden_size)
        return x
```

**DeepStack merger vs Final merger 的区别**：

| 方面 | DeepStack merger (×3) | Final merger (×1) |
|------|----------------------|-------------------|
| `use_postshuffle_norm` | `True` | `False` |
| LayerNorm 输入维度 | 4608 (merge 后) | 1152 (merge 前) |
| LayerNorm 时机 | reshape 到 `(-1, 4608)` **之后** | reshape **之前** |
| 功能 | 中间层特征投影 | 最终输出投影 |

**Shape 数据流**（以 `hidden_size=1152, spatial_merge_size=2, out_hidden_size=4096` 为例）：

```
输入: (196, 1152)
  ↓ view(-1, 4608)      ← 4 个相邻 patch 的特征拼接
(49, 4608)
  ↓ LayerNorm(4608)
(49, 4608)
  ↓ Linear(4608, 4608) + GELU
(49, 4608)
  ↓ Linear(4608, 4096)
(49, 4096)               ← 与 LLM hidden_size 一致
```

### 3.4 LLM Decoder 中的注入

**代码位置**：`Qwen3VLTextModel.forward()`，Line 818-837

#### 注入循环

```python
for layer_idx, decoder_layer in enumerate(self.layers):  # 32 层
    hidden_states = decoder_layer(hidden_states, ...)     # 正常 decoder layer
    # (B, L, 4096) → (B, L, 4096)

    # DeepStack 注入：仅在前 3 层
    if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
        hidden_states = self._deepstack_process(
            hidden_states,                         # (B, L, 4096)
            visual_pos_masks,                      # (B, L) bool
            deepstack_visual_embeds[layer_idx],    # (N_vis, 4096)
        )
```

| LLM Layer | 操作 |
|-----------|------|
| Layer 0 → 执行后 | **注入 ViT Layer 8 的特征** (deepstack_visual_embeds[0]) |
| Layer 1 → 执行后 | **注入 ViT Layer 16 的特征** (deepstack_visual_embeds[1]) |
| Layer 2 → 执行后 | **注入 ViT Layer 24 的特征** (deepstack_visual_embeds[2]) |
| Layer 3-31 | 正常 forward，无注入 |

#### `_deepstack_process` 方法 (Line 743-750)

```python
def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
    # hidden_states: (B, L, 4096)
    # visual_pos_masks: (B, L) — True 表示 vision token 位置
    # visual_embeds: (N_vis, 4096) — 来自 ViT 中间层的投影特征

    local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
    # hidden_states[visual_pos_masks, :] → (N_vis, 4096)  只取 vision token 位置
    # + visual_embeds                    → (N_vis, 4096)  残差加法
    # = local_this                       → (N_vis, 4096)

    hidden_states[visual_pos_masks, :] = local_this  # 写回原位置
    return hidden_states
    # (B, L, 4096) — 只有 vision token 位置的值被修改了
```

**关键特点**：
- **只修改 vision token 位置**的 hidden states，文本 token 不受影响
- **残差加法**（不是替换）：保留 LLM decoder layer 的输出，叠加 ViT 中间层信息
- **无可学习门控/权重**：直接做加法，完全靠训练让 ViT merger 学会输出合适尺度的特征

#### `visual_pos_masks` 的构建 (Line 1055-1077)

```python
# Qwen3VLModel.forward() 中
if image_mask is not None:
    image_mask = image_mask[..., 0]      # (B, L, 1) → (B, L)
    visual_pos_masks = image_mask        # True = 该位置是 image token

if video_mask is not None:
    video_mask = video_mask[..., 0]
    visual_pos_masks = image_mask | video_mask  # 图像 OR 视频位置
```

### 3.5 完整数据流图

```
                        ViT Encoder (27 层)
                        ===================

pixel_values: (N_patches, 1536)
    │
    ▼ patch_embed (Conv3d)
(N_patches, 1152)
    │
    ▼ + pos_embeds
(N_patches, 1152)
    │
    ├── ViT Block 0-7: (N_patches, 1152)
    │
    ├── ViT Block 8: (N_patches, 1152) ──→ deepstack_merger[0] ──→ (N_merged, out_hidden_size) ★
    │
    ├── ViT Block 9-15: (N_patches, 1152)
    │
    ├── ViT Block 16: (N_patches, 1152) ──→ deepstack_merger[1] ──→ (N_merged, out_hidden_size) ★
    │
    ├── ViT Block 17-23: (N_patches, 1152)
    │
    ├── ViT Block 24: (N_patches, 1152) ──→ deepstack_merger[2] ──→ (N_merged, out_hidden_size) ★
    │
    ├── ViT Block 25-26: (N_patches, 1152)
    │
    └── final merger ──→ (N_merged, out_hidden_size) → masked_scatter 到 input_embeds
         │
         ▼

                     LLM Decoder (32 层)
                     ===================

inputs_embeds: (B, L, 4096)
    │ [vision positions 已被 ViT final output 替换]
    │
    ├── LLM Layer 0 → hidden_states (B, L, 4096)
    │      └── _deepstack_process: h[vis_pos] += deepstack[0]   ← ViT Layer 8 特征
    │
    ├── LLM Layer 1 → hidden_states (B, L, 4096)
    │      └── _deepstack_process: h[vis_pos] += deepstack[1]   ← ViT Layer 16 特征
    │
    ├── LLM Layer 2 → hidden_states (B, L, 4096)
    │      └── _deepstack_process: h[vis_pos] += deepstack[2]   ← ViT Layer 24 特征
    │
    ├── LLM Layer 3-31 → 正常 forward (无注入)
    │
    └── LayerNorm → 输出: (B, L, 4096)
```

---

## 4. 两项技术的关联

MRoPE 和 DeepStack 从不同角度增强 Qwen3-VL 对视觉信息的处理能力：

| 维度 | MRoPE | DeepStack |
|------|-------|-----------|
| **解决的问题** | 图像/视频 token 的空间位置编码 | ViT 中间层特征被丢弃 |
| **作用位置** | 每个 Attention 层的 Q/K 旋转 | LLM 前 3 层 decoder 后的残差加法 |
| **影响的 token** | 所有 token（文本+视觉） | 仅 vision token |
| **信息类型** | 位置信息（相对距离、2D 网格） | 特征信息（纹理、结构、语义） |
| **可训练参数** | `inv_freq`（通常冻结） | 3 个 PatchMerger（各含 2 个 Linear） |

**协同效应**：MRoPE 让 LLM 知道每个 vision token 的空间位置，DeepStack 在此基础上在 LLM 的浅层注入多粒度视觉特征，使得 LLM 在早期就能获取丰富的视觉信息，而不必等到深层才"理解"图像内容。
