# Qwen3-VL 完整数据流：从原始输入到模型输出

本文档详细梳理 Qwen3-VL 从用户原始输入（图片、文本）经 Processor 预处理、ViT 编码、到 LLM Backbone 输出的完整流程。所有代码引用指向 HuggingFace Transformers 安装目录下的源文件。

> **路径缩写约定**：
> - `qwen3_vl/` = `transformers/models/qwen3_vl/`
> - `qwen2_vl/` = `transformers/models/qwen2_vl/`
> - `modular` = `qwen3_vl/modular_qwen3_vl.py`
> - `img_proc` = `qwen2_vl/image_processing_qwen2_vl.py`
> - `proc` = `qwen3_vl/processing_qwen3_vl.py`
> - `vid_proc` = `qwen3_vl/video_processing_qwen3_vl.py`

---

## 目录

1. [总览](#1-总览)
2. [阶段一：用户构造 Chat Messages](#2-阶段一用户构造-chat-messages)
3. [阶段二：Processor 处理](#3-阶段二processor-处理)
   - 3.1 [Image Processor — 图像预处理](#31-image-processor--图像预处理)
   - 3.2 [Video Processor — 视频预处理](#32-video-processor--视频预处理)
   - 3.3 [Token 占位符展开](#33-token-占位符展开)
   - 3.4 [文本 Tokenize](#34-文本-tokenize)
   - 3.5 [Processor 输出汇总](#35-processor-输出汇总)
4. [阶段三：Model Forward — 顶层入口](#4-阶段三model-forward--顶层入口)
5. [阶段四：ViT 视觉编码](#5-阶段四vit-视觉编码)
   - 5.1 [PatchEmbed — Conv3d 投影](#51-patchembed--conv3d-投影)
   - 5.2 [位置嵌入](#52-位置嵌入)
   - 5.3 [ViT Transformer Blocks](#53-vit-transformer-blocks)
   - 5.4 [PatchMerger — 空间合并](#54-patchmerger--空间合并)
6. [阶段五：Vision Embedding 注入 LLM](#6-阶段五vision-embedding-注入-llm)
7. [阶段六：LLM Backbone Forward](#7-阶段六llm-backbone-forward)
8. [阶段七：LM Head 输出](#8-阶段七lm-head-输出)
9. [端到端 Tensor Shape 追踪表](#9-端到端-tensor-shape-追踪表)
10. [关键公式汇总](#10-关键公式汇总)

---

## 1. 总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         用户输入                                        │
│  images: [PIL.Image, ...]                                               │
│  text: "Describe this image: <|image_pad|>"                             │
│  (或 chat messages 格式)                                                │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│               Processor (Qwen3VLProcessor)                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│  │ Image Processor   │  │ Video Processor   │  │ Tokenizer            │   │
│  │ smart_resize      │  │ smart_resize      │  │ apply_chat_template  │   │
│  │ rescale/normalize │  │ rescale/normalize │  │ 占位符展开            │   │
│  │ patch reshape     │  │ patch reshape     │  │ tokenize             │   │
│  └────────┬─────────┘  └────────┬─────────┘  └──────────┬───────────┘   │
│           │                      │                        │               │
│           ▼                      ▼                        ▼               │
│  pixel_values          pixel_values_videos         input_ids             │
│  image_grid_thw        video_grid_thw              attention_mask        │
└──────────┬───────────────────────┬────────────────────────┬──────────────┘
           │                       │                        │
           ▼                       ▼                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                    Model Forward                                         │
│  ┌─────────────┐   ┌────────────────────┐   ┌────────────────────────┐  │
│  │ ViT Encoder  │   │ Embedding 替换      │   │ LLM Backbone           │  │
│  │ Conv3d       │   │ masked_scatter      │   │ 32 Decoder Layers      │  │
│  │ 27 Blocks    │   │ get_rope_index      │   │ + MRoPE + DeepStack    │  │
│  │ PatchMerger  │   │                     │   │ + Causal Mask          │  │
│  └──────┬──────┘   └─────────┬───────────┘   └──────────┬────────────┘  │
│         │                     │                           │               │
│         └─────────────────────┘                           ▼               │
│                                                   LM Head → logits       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 阶段一：用户构造 Chat Messages

用户以 Qwen3-VL 的 chat 格式构造输入（以 Alpamayo 为例）：

```python
# helper.py:77-142, create_message()
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a driving assistant..."}]},
    {"role": "user", "content": [
        {"type": "text", "text": "Front camera: "},
        {"type": "text", "text": "frame 0 "},
        {"type": "image", "image": tensor(3, H, W)},   # ← 每帧作为独立图像
        {"type": "text", "text": "frame 1 "},
        {"type": "image", "image": tensor(3, H, W)},
        ...
        {"type": "text", "text": "output the chain-of-thought reasoning..."},
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "<|cot_start|>"}]},
]
```

此时图像还是原始的 PIL.Image 或 Tensor，文本中没有展开 `<|image_pad|>` 的数量。

---

## 3. 阶段二：Processor 处理

**入口**：`Qwen3VLProcessor.__call__()` 或 `apply_chat_template()`
**代码位置**：`proc:114-247`

Processor 内部分三路处理：Image Processor、Video Processor、Tokenizer。

### 3.1 Image Processor — 图像预处理

**代码位置**：`img_proc:83-297`（`Qwen2VLImageProcessor`）

Qwen3-VL 复用 Qwen2-VL 的 image processor，通过 `preprocessor_config.json` 配置参数。

#### Step 1: `smart_resize()` — 动态分辨率调整

**代码位置**：`img_proc:54-80`

```python
def smart_resize(height, width, factor=28, min_pixels=56*56, max_pixels=28*28*1280):
    # 1) 对齐到 factor 的倍数
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    # 2) 超出 max_pixels → 按比例缩小
    if h_bar * w_bar > max_pixels:
        beta = sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor(height / beta / factor) * factor)
        w_bar = max(factor, floor(width / beta / factor) * factor)

    # 3) 不足 min_pixels → 按比例放大
    elif h_bar * w_bar < min_pixels:
        beta = sqrt(min_pixels / (height * width))
        h_bar = ceil(height * beta / factor) * factor
        w_bar = ceil(width * beta / factor) * factor

    return h_bar, w_bar
```

**关键参数**：
- `factor = patch_size * merge_size`（确保 resize 后的 H, W 能被 patch 化后整除 merge_size）
- `min_pixels`, `max_pixels`：从 processor config 读取，控制分辨率范围

**调用点**：`img_proc:248-254`

```python
resized_height, resized_width = smart_resize(
    height, width,
    factor=patch_size * merge_size,
    min_pixels=size["shortest_edge"],
    max_pixels=size["longest_edge"],
)
```

#### Step 2: Resize + Rescale + Normalize

**代码位置**：`img_proc:256-265`

```python
image = resize(image, size=(resized_height, resized_width), resample=resample)
# 如果 do_rescale: image = image / 255.0
# 如果 do_normalize: image = (image - mean) / std
#   mean = [0.48145466, 0.4578275, 0.40821073]
#   std  = [0.26862954, 0.26130258, 0.27577711]
```

#### Step 3: 时间维度 Padding

**代码位置**：`img_proc:270-277`

单张图片只有 1 帧，但 Conv3d 的 `temporal_patch_size=2` 要求时间维度是 2 的倍数：

```python
patches = np.array([processed_image])  # shape: (1, C, H, W)

if patches.shape[0] % temporal_patch_size != 0:    # 1 % 2 != 0 → True
    repeats = np.repeat(patches[-1][np.newaxis], 1, axis=0)  # 复制最后一帧
    patches = np.concatenate([patches, repeats], axis=0)
# patches shape: (2, C, H_resized, W_resized)
```

#### Step 4: Patch 化 — 关键 reshape 操作

**代码位置**：`img_proc:278-295`

这一步将像素图像重排为 ViT 的 patch 输入格式：

```python
grid_t = patches.shape[0] // temporal_patch_size    # 2 // 2 = 1
grid_h = resized_height // patch_size               # e.g., 448 // 14 = 32
grid_w = resized_width // patch_size                # e.g., 448 // 14 = 32

# 9维 reshape: 把空间维度按 merge_size 分组
patches = patches.reshape(
    grid_t,                    # 时间网格
    temporal_patch_size,       # 2 (时间 patch)
    channel,                   # 3 (RGB)
    grid_h // merge_size,      # 空间 H 方向分组数
    merge_size,                # 2 (空间合并)
    patch_size,                # 14 (patch 像素高)
    grid_w // merge_size,      # 空间 W 方向分组数
    merge_size,                # 2 (空间合并)
    patch_size,                # 14 (patch 像素宽)
)
# shape 示例: (1, 2, 3, 16, 2, 14, 16, 2, 14)

# transpose 重排维度: 把 merge 的 patch 和像素维度放到一起
patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
# shape: (grid_t, grid_h//ms, grid_w//ms, ms, ms, C, temp_ps, ps, ps)
# 示例: (1, 16, 16, 2, 2, 3, 2, 14, 14)

# 最终展平
flatten_patches = patches.reshape(
    grid_t * grid_h * grid_w,
    channel * temporal_patch_size * patch_size * patch_size
)
# shape: (1*32*32, 3*2*14*14) = (1024, 1176)
```

**每个 patch 包含的像素**：`C × temporal_patch_size × patch_size × patch_size` = `3 × 2 × 14 × 14` = **1176** 维

> **注意**：transpose 的关键作用是把原本分散的 2×2 空间 merge 区域内的 patch 排列到相邻位置。这样在 ViT 最后的 PatchMerger 中做 `view(-1, hidden_size * merge_size²)` 时，相邻的 4 个 patch 恰好是空间上相邻的 2×2 区域。

#### 输出

```python
return flatten_patches, (grid_t, grid_h, grid_w)
# flatten_patches shape: (grid_t * grid_h * grid_w, C * temporal_ps * ps * ps)
# grid_thw:             (grid_t, grid_h, grid_w) — 用于后续计算 token 数量和 RoPE
```

**多张图片**时，所有图片的 `flatten_patches` 在 batch 维度上拼接：

```python
pixel_values = np.concatenate([fp1, fp2, ...], axis=0)
# shape: (total_patches, feature_dim)
image_grid_thw = [[1, H1', W1'], [1, H2', W2'], ...]
# shape: (num_images, 3)
```

### 3.2 Video Processor — 视频预处理

**代码位置**：`vid_proc:87-273`（`Qwen3VLVideoProcessor`）

与图像类似，但 `patch_size=16`（不同于 image 的 14），且处理多帧。

| 参数 | Image Processor | Video Processor |
|------|----------------|-----------------|
| `patch_size` | 14 | 16 |
| `factor` | 14×2=28 | 16×2=32 |
| `temporal_patch_size` | 2 | 2 |
| `merge_size` | 2 | 2 |
| 每个 patch 的维度 | 3×2×14×14=1176 | 3×2×16×16=1536 |

视频的 grid_thw 中 T 可以 >1（`T = num_frames // temporal_patch_size`）。

### 3.3 Token 占位符展开

**代码位置**：`proc:186-234`

Processor 在 `apply_chat_template()` 生成文本后，需要将模板中的单个 `<|image_pad|>` 或 `<|video_pad|>` 展开为正确数量的重复 token。

#### 图像 token 展开 (`proc:186-194`)

```python
merge_length = self.image_processor.merge_size ** 2    # 2² = 4

for each image:
    num_image_tokens = image_grid_thw[index].prod() // merge_length
    # 例: grid_thw = [1, 32, 32] → 1*32*32 / 4 = 256 个 token

    text[i] = text[i].replace("<|image_pad|>", "<|placeholder|>" * 256, 1)
    # 最后再把 <|placeholder|> 统一替换回 <|image_pad|>

text[i] = text[i].replace("<|placeholder|>", "<|image_pad|>")
```

**token 数量公式**：

```
num_image_tokens = (grid_t × grid_h × grid_w) / merge_size²
```

#### 视频 token 展开 (`proc:196-234`)

视频会为每帧插入时间戳 text token：

```python
frame_seqlen = video_grid_thw[index][1:].prod() // merge_length
# 例: grid_thw = [4, 32, 32] → 32*32 / 4 = 256 tokens per frame

for frame_idx in range(video_grid_thw[index][0]):   # T=4 帧
    curr_time = curr_timestamp[frame_idx]
    video_placeholder += f"<{curr_time:.1f} seconds>"
    video_placeholder += "<|vision_start|>" + "<|placeholder|>" * 256 + "<|vision_end|>"
```

展开后的文本结构：

```
<0.5 seconds><|vision_start|><|video_pad|>×256<|vision_end|>
<2.5 seconds><|vision_start|><|video_pad|>×256<|vision_end|>
```

### 3.4 文本 Tokenize

**代码位置**：`proc:236-239`

```python
text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
# 输出: {"input_ids": (B, L), "attention_mask": (B, L)}
```

其中 `<|image_pad|>` 和 `<|video_pad|>` 是 tokenizer 词表中的特殊 token，各有唯一的 token_id。

### 3.5 Processor 输出汇总

**代码位置**：`proc:247`

```python
return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type="pt")
```

| 输出 Tensor | Shape | 含义 |
|------------|-------|------|
| `input_ids` | `(B, L)` | token 序列，含文本 token + `<\|image_pad\|>` × N + `<\|video_pad\|>` × M |
| `attention_mask` | `(B, L)` | 1=有效, 0=padding |
| `pixel_values` | `(total_img_patches, feat_dim)` | 所有图片的 patch 特征拼接 |
| `image_grid_thw` | `(num_images, 3)` | 每张图片的 (T, H, W) 网格 |
| `pixel_values_videos` | `(total_vid_patches, feat_dim)` | 所有视频的 patch 特征拼接 |
| `video_grid_thw` | `(num_videos, 3)` | 每个视频的 (T, H, W) 网格 |

**具体示例**（1 张 448×448 图片，patch_size=14, merge_size=2）：

```
pixel_values:     (1024, 1176)    — 1*32*32=1024 patches, 3*2*14*14=1176 dim
image_grid_thw:   [[1, 32, 32]]
num_image_tokens: 1024 / 4 = 256  — 在 input_ids 中占 256 个 <|image_pad|>
```

---

## 4. 阶段三：Model Forward — 顶层入口

### `Qwen3VLForConditionalGeneration.forward()`

**代码位置**：`modular:1153-1211`

```python
def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, ...):
    # 1) 调用内部 model
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
        ...
    )
    hidden_states = outputs.last_hidden_state   # (B, L, 4096)

    # 2) LM Head 投影到词表
    logits = self.lm_head(hidden_states)         # (B, L, vocab_size)
    return logits
```

### `Qwen3VLModel.forward()` — 核心调度

**代码位置**：`modular:1010-1141`

这是核心调度函数，串联 ViT 编码、embedding 替换、position_id 计算和 LLM forward。按顺序执行：

```python
def forward(self, input_ids, pixel_values, image_grid_thw, ...):
    # Step 1: 文本 embedding
    inputs_embeds = self.get_input_embeddings()(input_ids)
    # (B, L) → (B, L, hidden_size=4096)

    # Step 2: ViT 编码图像
    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0)
        # image_embeds: (total_merged_tokens, out_hidden_size)

        # Step 3: 找到 <|image_pad|> 位置并替换
        image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # Step 4: 同样处理视频 (如有)
    if pixel_values_videos is not None:
        ...  # 同图像流程

    # Step 5: 构建 visual_pos_masks 和 deepstack features
    visual_pos_masks = image_mask[..., 0]  # (B, L) bool
    deepstack_visual_embeds = deepstack_image_embeds  # list of 3 tensors

    # Step 6: 计算 3D RoPE position_ids
    position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
    # position_ids: (3, B, L),  rope_deltas: (B, 1)

    # Step 7: 调用 LLM backbone
    outputs = self.language_model(
        inputs_embeds=inputs_embeds,       # (B, L, 4096)
        position_ids=position_ids,          # (3, B, L)
        attention_mask=attention_mask,       # (B, L) → 内部转为 4D causal
        visual_pos_masks=visual_pos_masks,  # (B, L) bool
        deepstack_visual_embeds=deepstack_visual_embeds,
        ...
    )
    return outputs
```

---

## 5. 阶段四：ViT 视觉编码

### `Qwen3VLVisionModel.forward()`

**代码位置**：`modular:676-726`

### 5.1 PatchEmbed — Conv3d 投影

**代码位置**：`modular:341-350`（`Qwen3VLVisionPatchEmbed`）

```python
class Qwen3VLVisionPatchEmbed(PatchEmbed):
    # patch_size = 16, temporal_patch_size = 2, in_channels = 3, embed_dim = 1152
    self.proj = nn.Conv3d(3, 1152, kernel_size=[2, 16, 16], stride=[2, 16, 16], bias=True)
```

**PatchEmbed.forward()**（`qwen2_vl/modeling_qwen2_vl.py:246-249`）：

```python
def forward(self, hidden_states):
    hidden_states = hidden_states.view(-1, 3, 2, 16, 16)
    # (N_patches, 1176或1536) → (N_patches, C=3, T=2, H=16, W=16)
    hidden_states = self.proj(hidden_states).view(-1, self.embed_dim)
    # Conv3d: (N, 3, 2, 16, 16) → (N, 1152, 1, 1, 1) → view → (N, 1152)
    return hidden_states
```

**Tensor 变化**：

```
pixel_values:  (N_patches, C*T*H*W)    例: (1024, 1176)
     ↓ view
                (N_patches, 3, 2, 16, 16)   → 5D for Conv3d
     ↓ Conv3d(kernel=[2,16,16], stride=[2,16,16])
                (N_patches, 1152, 1, 1, 1)  → 3D卷积完全覆盖
     ↓ view
                (N_patches, 1152)            → ViT 序列
```

Conv3d 的 kernel 和 stride 完全等于 patch 尺寸，所以每个 patch 被投影为一个 1152 维向量。这实际上等价于一个线性投影层。

### 5.2 位置嵌入

**代码位置**：`modular:689-698`

```python
# (1) 绝对位置嵌入 — 可学习 Embedding
pos_embeds = self.fast_pos_embed_interpolate(grid_thw)   # (N_patches, 1152)
hidden_states = hidden_states + pos_embeds                # 残差加

# (2) ViT 内部旋转位置嵌入 (2D 空间 RoPE, 非 LLM 的 3D MRoPE)
rotary_pos_emb = self.rot_pos_emb(grid_thw)
emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
position_embeddings = (emb.cos(), emb.sin())
# 用于 ViT Attention 中的 Q/K 旋转
```

**ViT 的 RoPE 是 2D 空间的（只有 H, W），不同于 LLM 的 3D MRoPE（T, H, W）。**

### 5.3 ViT Transformer Blocks

**代码位置**：`modular:710-722`

```python
# cu_seqlens: Flash Attention 的变长序列边界
cu_seqlens = torch.repeat_interleave(
    grid_thw[:, 1] * grid_thw[:, 2],   # 每个时间步的 H*W
    grid_thw[:, 0]                       # 重复 T 次
).cumsum(dim=0)
cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
# 例: grid_thw=[[1,32,32]] → cu_seqlens = [0, 1024]

deepstack_feature_lists = []
for layer_num, blk in enumerate(self.blocks):    # 27 层
    hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, position_embeddings=...)
    # (N_patches, 1152) → (N_patches, 1152)

    if layer_num in self.deepstack_visual_indexes:  # [8, 16, 24]
        feat = self.deepstack_merger_list[...](hidden_states)
        deepstack_feature_lists.append(feat)
        # feat: (N_merged, out_hidden_size)
```

**cu_seqlens 的作用**：

ViT 使用 Flash Attention 2 的 variable-length mode。多张图片的 patch 拼接在一起形成一个长序列，`cu_seqlens` 标记每张图片的边界，使 attention 只在同一张图片内部计算。

```
image_1: 1024 patches | image_2: 256 patches
cu_seqlens = [0, 1024, 1280]
→ patches 0:1024 属于 image_1, patches 1024:1280 属于 image_2
→ 两张图片之间没有 attention 交互
```

**每个 ViT Block 内部结构**（`modular:379-386`）：

```
hidden_states
  ↓ norm1 (LayerNorm)
  ↓ attn (VisionAttention + 2D spatial RoPE + Flash Attention w/ cu_seqlens)
  ↓ + residual
  ↓ norm2 (LayerNorm)
  ↓ mlp (Linear(1152→4304) → GELU → Linear(4304→1152))
  ↓ + residual
→ hidden_states   shape 不变: (N_patches, 1152)
```

### 5.4 PatchMerger — 空间合并

**代码位置**：`modular:357-370`（`Qwen3VLVisionPatchMerger`）

在 ViT 的全部 27 层结束后，以及在 DeepStack 提取点，PatchMerger 将空间上相邻的 `merge_size² = 4` 个 patch 合并为 1 个 token：

```python
class Qwen3VLVisionPatchMerger(nn.Module):
    # hidden_size = 1152 * (2²) = 4608
    self.norm = nn.LayerNorm(4608 or 1152)
    self.linear_fc1 = nn.Linear(4608, 4608)
    self.act_fn = nn.GELU()
    self.linear_fc2 = nn.Linear(4608, out_hidden_size)  # → LLM 维度

    def forward(self, x):
        # x: (N_patches, 1152)
        x = self.norm(x.view(-1, 4608) if postshuffle else x).view(-1, 4608)
        # → (N_patches/4, 4608)   — 4 个 patch 的特征拼接
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        # → (N_patches/4, out_hidden_size)
        return x
```

**Tensor 变化**：

```
输入: (1024, 1152)    — 32×32 = 1024 个 patch
  ↓ view(-1, 4608)    — 4 个相邻 patch 拼接: 1152 × 4 = 4608
(256, 4608)
  ↓ LayerNorm
  ↓ Linear(4608→4608) + GELU
  ↓ Linear(4608→out_hidden_size)
(256, out_hidden_size)  — 合并后 256 = 1024/4 个 token
```

#### 最终 Vision Model 输出 (`modular:724-726`)

```python
hidden_states = self.merger(hidden_states)  # 最终输出
return hidden_states, deepstack_feature_lists
```

| 输出 | Shape | 说明 |
|------|-------|------|
| `hidden_states` | `(N_merged, out_hidden_size)` | 主 vision embedding |
| `deepstack_feature_lists` | list of 3 × `(N_merged, out_hidden_size)` | 来自 ViT Layer 8/16/24 |

---

## 6. 阶段五：Vision Embedding 注入 LLM

### Step 1: 按图片拆分 (`modular:989-990`)

```python
split_sizes = (image_grid_thw.prod(-1) // spatial_merge_size**2).tolist()
image_embeds = torch.split(image_embeds, split_sizes)
# 例: 2张图, split_sizes = [256, 64]
# → tuple of (256, out_hidden_size), (64, out_hidden_size)
```

### Step 2: 拼接所有图片 (`modular:1041`)

```python
image_embeds = torch.cat(image_embeds, dim=0)
# → (total_image_tokens, out_hidden_size)
```

### Step 3: 找到占位符位置 (`modular:1042-1044`)

```python
image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds, image_features=image_embeds)
```

**get_placeholder_mask()**（继承自 `qwen2_5_vl`）：

```python
special_image_mask = (input_ids == self.config.image_token_id)  # (B, L) bool
# 扩展到 3D 以匹配 inputs_embeds 的 shape
special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds)
# (B, L) → (B, L, hidden_size)
```

### Step 4: masked_scatter 替换 (`modular:1045`)

```python
inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

**masked_scatter 操作**：
1. `inputs_embeds`：`(B, L, hidden_size)` — 所有 token 的 embedding
2. `image_mask`：`(B, L, hidden_size)` — `True` 位置是 `<|image_pad|>` token
3. `image_embeds`：`(total_image_tokens, out_hidden_size)` — ViT 编码结果
4. 将 `image_embeds` 的值按顺序填入 `inputs_embeds` 中 `mask=True` 的位置

**效果**：`inputs_embeds` 中原本的 `<|image_pad|>` 词嵌入被替换为 ViT 编码后的视觉特征。

### Step 5: 计算 3D MRoPE Position IDs (`modular:1102-1123`)

```python
position_ids, rope_deltas = self.get_rope_index(
    input_ids, image_grid_thw, video_grid_thw, attention_mask
)
# position_ids: (3, B, L)  — [T, H, W] 三维位置
# rope_deltas:  (B, 1)     — 位置偏移量
```

详见 [qwen3vl.md 第 2.2 节](qwen3vl.md#22-3d-position-ids-的构建)。

---

## 7. 阶段六：LLM Backbone Forward

### `Qwen3VLTextModel.forward()`

**代码位置**：`modular:754-844`

```python
def forward(self, inputs_embeds, position_ids, attention_mask,
            visual_pos_masks, deepstack_visual_embeds, ...):

    # 1) 准备 3D MRoPE 位置编码
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    # → (cos, sin), 各 (B, L, head_dim=128)

    # 2) 创建 4D 因果掩码
    attention_mask = create_causal_mask(config, inputs_embeds, attention_mask, ...)
    # → (B, 1, L, KV_len)

    hidden_states = inputs_embeds   # (B, L, 4096)

    # 3) 32 层 Decoder + DeepStack
    for layer_idx, decoder_layer in enumerate(self.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            ...
        )
        # (B, L, 4096) → (B, L, 4096)

        # DeepStack 注入: 仅前 3 层
        if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
            hidden_states = self._deepstack_process(
                hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx]
            )

    # 4) 最终 LayerNorm
    hidden_states = self.norm(hidden_states)
    # → (B, L, 4096)
```

**每个 Decoder Layer 内部**（`modular:496-521`）：

```
hidden_states (B, L, 4096)
  ↓ input_layernorm (RMSNorm)
  ↓ self_attn:
  │   Q = q_norm(q_proj(x))  → (B, 32, L, 128)
  │   K = k_norm(k_proj(x))  → (B, 32, L, 128)
  │   V = v_proj(x)          → (B, 32, L, 128)
  │   Q, K = apply_rotary_pos_emb(Q, K, cos, sin)  ← MRoPE
  │   attn_output = attention(Q, K, V, causal_mask)
  │   output = o_proj(attn_output)  → (B, L, 4096)
  ↓ + residual
  ↓ post_attention_layernorm (RMSNorm)
  ↓ mlp:
  │   gate = gate_proj(x)    → (B, L, 22016)
  │   up   = up_proj(x)      → (B, L, 22016)
  │   x = silu(gate) * up
  │   x = down_proj(x)       → (B, L, 4096)
  ↓ + residual
→ hidden_states (B, L, 4096)
```

**DeepStack 注入**详见 [qwen3vl.md 第 3.4 节](qwen3vl.md#34-llm-decoder-中的注入)。

---

## 8. 阶段七：LM Head 输出

**代码位置**：`modular:1196-1198`

```python
hidden_states = outputs.last_hidden_state    # (B, L, 4096)
logits = self.lm_head(hidden_states)          # Linear(4096, vocab_size)
# → (B, L, 151936)
```

---

## 9. 端到端 Tensor Shape 追踪表

以 1 张 448×448 图片 + 100 个文本 token 为例（`patch_size=14, merge_size=2`）：

| 阶段 | 操作 | Tensor | Shape |
|------|------|--------|-------|
| **Processor** | 原始图片 | image | `(448, 448, 3)` |
| | smart_resize | resized | `(448, 448, 3)` |
| | rescale+normalize | normalized | `(3, 448, 448)` float |
| | 时间 pad (1→2) | patches | `(2, 3, 448, 448)` |
| | patch reshape+flatten | pixel_values | `(1024, 1176)` |
| | grid_thw | image_grid_thw | `[[1, 32, 32]]` |
| | token 计算 | num_image_tokens | `1024/4 = 256` |
| | tokenize | input_ids | `(1, 356)` = 100 text + 256 image |
| **ViT** | Conv3d patch embed | hidden_states | `(1024, 1152)` |
| | + pos_embed | hidden_states | `(1024, 1152)` |
| | ViT Block ×27 | hidden_states | `(1024, 1152)` |
| | DeepStack @8,16,24 | deepstack[i] | `(256, out_hidden_size)` each |
| | Final PatchMerger | image_embeds | `(256, out_hidden_size)` |
| **Embedding 替换** | text embedding | inputs_embeds | `(1, 356, 4096)` |
| | masked_scatter | inputs_embeds | `(1, 356, 4096)` — 256 个位置被替换 |
| | get_rope_index | position_ids | `(3, 1, 356)` |
| **LLM** | create_causal_mask | attention_mask | `(1, 1, 356, 356)` |
| | MRoPE | cos, sin | `(1, 356, 128)` each |
| | Decoder ×32 | hidden_states | `(1, 356, 4096)` |
| | DeepStack @0,1,2 | hidden_states | `(1, 356, 4096)` — 仅 256 个 vision 位置修改 |
| | Final LayerNorm | hidden_states | `(1, 356, 4096)` |
| **LM Head** | Linear | logits | `(1, 356, 151936)` |

---

## 10. 关键公式汇总

### 分辨率计算

```
resized_h = round(h / factor) * factor,  factor = patch_size × merge_size
resized_w = round(w / factor) * factor
约束: min_pixels ≤ resized_h × resized_w ≤ max_pixels
```

### Grid 计算

```
grid_t = num_frames / temporal_patch_size        (图片恒为 1)
grid_h = resized_h / patch_size
grid_w = resized_w / patch_size
```

### Token 数量

```
num_patches  = grid_t × grid_h × grid_w          (ViT 处理的 patch 数)
num_tokens   = num_patches / merge_size²          (LLM 中的 vision token 数)
```

### Patch 特征维度

```
feat_dim = C × temporal_patch_size × patch_size × patch_size
         = 3 × 2 × patch_size²
```

### PatchMerger 维度变化

```
输入: (num_patches, vit_hidden_size)          = (1024, 1152)
合并: (num_patches/4, vit_hidden_size × 4)   = (256, 4608)
投影: (num_patches/4, out_hidden_size)         = (256, llm_hidden_size)
```
