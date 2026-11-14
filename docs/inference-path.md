# 第二阶段：图片与文本融合及文本生成

整理日期：2026-11-14。形状采集由用户于 2026-11-13 亲自运行，原始记录为 [m2-shapes-20261113-233039-36b4b251.json](../results/evidence/m2-shapes.json)，状态为 completed。本文对应上游提交 `041d528e037a8a373c3c8aa7278a3ba3bd26b6dc` 及当前本地教学注释版本，不代表训练、质量评测或性能测试。

## 一、图片转化为张量，并与文本张量合并

### 1. 加载模型与配套组件

入口是 `eval_vlm.py` 的 `init_model`。`MiniMindVLM` 包含语言模型、SigLIP 视觉编码器和视觉投影层；tokenizer 使用本地 `model/`，预处理器与视觉编码器配套。语言与投影权重从 `out/sft_vlm_768.pth` 加载，视觉编码器从 `model/siglip2-base-p32-256-ve/` 加载。

`model.eval()` 切换推理状态；CUDA 路径中的 `model.half()` 将模型转换为 FP16，再通过 `.to(device)` 移到 GPU。加载现有权重不等于训练，推理不会更新权重。

### 2. 图片读取与预处理

`eval_vlm.py` 先过滤图片扩展名，再通过 `Image.open(...).convert('RGB')` 统一为 RGB 三通道。`MiniMindVLM.image2tensor(image, preprocess)` 调用配套 processor，完成尺寸、像素数值等预处理，返回包含 PyTorch 张量的字典。

```python
pixel_values = {
    k: v.to(args.device)
    for k, v in MiniMindVLM.image2tensor(image, preprocess).items()
}
```

`k` 是字典键，`v` 是对应张量；`.to(args.device)` 在这里用于迁移设备，没有显式指定 dtype。本次原图为 640×640，预处理后的实际输入为 `[1, 3, 256, 256]`、float32、cuda:0，四个轴依次表示 batch、通道、高、宽。

### 3. 文本编码与图片位置预留

原始问题是 `<image>\n请描述这张图中的主要物体和场景。`。`prompt.replace(...)` 将 `<image>` 替换成 64 个连续的 `<|image_pad|>`，其 token ID 为 12，数量来自本版本的 `image_token_len=64`。

`messages` 是带 role/content 的对话列表。`apply_chat_template(..., tokenize=False)` 先生成格式化字符串，包括用户、助手边界等标记；下一句 `tokenizer(..., return_tensors='pt')` 才真正得到 token ID 和 attention mask，并迁移到 GPU。

本次 `input_ids` 与 `attention_mask` 均为 `[1, 93]`、int64。93 个位置中有 64 个图片占位符，剩余 29 个包括问题和模板标记，不能把它们都当作问题正文。attention mask 用于标识输入位置是否有效，与模型内部的因果注意力约束作用不同。

### 4. 文本 ID 变成嵌入

`MiniMindVLM.forward` 中执行：

```python
hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))
```

每个 token ID 查表得到一个 768 维向量，实际形状由 `[1, 93]` 变为 `[1, 93, 768]`、float16。此时 64 个图片位置仍是占位符的普通嵌入，尚未包含图像内容。

### 5. 视觉编码与投影

本次字典中的图片张量是四维，走单图分支：

```python
vision_tensors = self.vision_proj(
    MiniMindVLM.get_image_embeddings(pixel_values, self.vision_encoder)
)
```

`get_image_embeddings` 在 `torch.no_grad()` 下调用视觉编码器，取 `last_hidden_state`。本版本输入 256×256、patch 大小 32，对应 8×8=64 个 patch，实际得到 `[1, 64, 768]`、float16。经视觉 Transformer 后，一个位置的特征可以融合其他 patch 的信息，不宜理解成只描述独立小方块。

`vision_proj` 是 `LayerNorm → Linear → GELU → Linear`，输出仍为 `[1, 64, 768]`。它处理的输入本来就是张量，作用是变换特征表示，使其适配语言模型；视觉和语言维度都为 768，不意味着可以省略已训练的投影层。

### 6. 替换占位符嵌入，实现融合

`count_vision_proj(tokens, h, vision_tensors, seqlen)` 根据 token ID 找到连续的图片占位符区间，将该区间对应的文本嵌入替换为视觉向量。核心拼接是：

```python
hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
```

`start:i` 是占位符区间，`k` 是当前图片索引。函数保留区间前后的文本嵌入，在中间放入视觉向量。单图的三维视觉特征会先增加一个图片数量轴，便于统一按图片索引处理。

```text
融合前：29 个其他 token 的嵌入 + 64 个占位符嵌入，共 93 个位置
融合后：29 个其他 token 的嵌入 + 64 个视觉向量，共 93 个位置
```

**64 个视觉向量替换了提前预留的 64 个位置，因此不会变成 157 个位置。** 实测 merge.before、merge.after 和第一层 Transformer 输入均为 `[1, 93, 768]`。形状记录证明长度一致；向量替换机制由源码确认，不能仅靠形状相同判断数值是否改变。

## 二、模型处理张量并生成文本

### 1. 找到实际生成循环

```text
eval_vlm.py 或 scripts/trace_vlm_shapes.py 调用 model.generate
  → model/model_vlm.py：MiniMindVLM.generate
  → super().generate
  → model/model_minimind.py：MiniMindForCausalLM.generate
  → self.forward：实际执行 MiniMindVLM.forward
```

VLM 继承语言模型的生成循环，但 `self` 仍是 VLM 实例，因此循环中的 `self.forward` 会调用包含图片融合逻辑的 VLM 前向传播，不能只看语言模型的普通文本 forward 来解释本次路径。

### 2. 首次前向：处理完整提示序列

第一次缓存为空，`past_len=0`，生成循环将完整的 93 个 token 传给 forward。VLM 完成文本嵌入和图片融合，取本次位置对应的 RoPE 位置编码，再依次执行 8 层 Transformer。

每层读取当前 `hidden_states` 和本层旧缓存，返回更新后的 `hidden_states` 与本层缓存 `present`；`presents` 收集所有层的缓存。实际层类型为 `MiniMindBlock`，包含注意力、残差、归一化与前馈计算。推理中 `use_cache=True`，旧位置的 K、V 可被后续生成复用。

最后经过 `self.model.norm`，由 `lm_head` 将每个位置的 768 维表示映射到 6400 维词表分数。本次 logits 实测为 `[1, 93, 6400]`。logits 是分数，不是已归一化的概率。

### 3. 从最后位置选择下一个 token

实际代码在 `model_minimind.py` 的 generate 中：

```python
logits = outputs.logits[:, -1, :] / temperature
```

只取最后位置对下一个 token 的预测，本次形状成为 `[1, 6400]`。原函数还保留重复惩罚、top-k 和 top-p 筛选分支；M2 设置 repetition_penalty=1、top_k=0、top_p=1、temperature=1，不启用这些分数调整或筛选。

```python
next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
```

`do_sample=True` 时从概率分布采样，`False` 时选择最大分数的 token ID。M2 使用 `False`，第一次选出 394。M1 上游入口使用采样，不能将两次生成策略混为一谈。

### 4. 追加 token 与复用 KV cache

生成循环把新 ID 追加到 `input_ids`，并将 `outputs.past_key_values` 保存为下次缓存。第一次处理的 93 个位置已经产生缓存，而刚预测出的新 token 尚未经过前向计算。

下一轮通过 `input_ids[:, past_len:]` 只取未缓存的部分。因此第二次输入长度为 1、旧缓存长度为 93，而不是把完整序列重新计算一遍。

```text
第一次：93 个输入位置 → 预测 ID 394 → 保存前 93 个位置的缓存
第二次：输入 ID 394，利用 93 个旧位置的缓存 → 预测 ID 2370
```

新 token 的 Q 可以关注缓存中旧位置的 K、V，以及本次新位置的 K、V，所以只输入一个 token 仍能利用图片和问题的信息。缓存保存的是各层 K、V，不是原始图片或最终文字。

VLM 用 `start_pos == 0` 判断是否执行视觉编码和融合。本次第二轮 start_pos=93，不再计算视觉编码器和投影层，实测日志中也没有第二组视觉观察点。这个结论依赖本次启用了缓存；关闭缓存时不能照搬。

第二轮文本嵌入和第一层输入为 `[1, 1, 768]`，logits 为 `[1, 1, 6400]`。第二个 token 被预测后达到本次生成上限，不需要再把它送入模型进行第三次前向。

### 5. 停止并解码

生成循环最多执行 `max_new_tokens` 次；若所有样本生成 EOS，可提前结束。返回的 `generated_ids` 包含原提示 ID 和新 ID，采集脚本先切去提示部分，再调用 tokenizer.decode。

```python
new_ids = generated[0, inputs.input_ids.shape[1]:].tolist()
tokenizer.decode(new_ids, skip_special_tokens=True)
```

本次新 ID 为 `[394, 2370]`，共同解码为“这张”，停止原因是 length_limit，即达到 2 个新 token 上限。token 不保证对应一个汉字，也可能对应词、标点或文字片段。

原生 eval_vlm.py 使用 TextStreamer 边生成边显示文字；本次采集脚本在生成结束后解码新 ID，两者是不同的显示入口。

## 三、实测形状与采集方式

| 观察位置 | 首次前向 | 第二次前向 |
| --- | --- | --- |
| input_length / past_length | 93 / 0 | 1 / 93 |
| 图片预处理输入 | [1,3,256,256]，float32 | 复用原输入，不再次视觉编码 |
| 文本嵌入 | [1,93,768] | [1,1,768] |
| 视觉编码输出 | [1,64,768] | 未执行 |
| 投影输出 | [1,64,768] | 未执行 |
| 融合后 / 第一层输入 | [1,93,768] | [1,1,768]，仅第一层输入 |
| logits | [1,93,6400] | [1,1,6400] |

除图片预处理输入与整数 ID 外，上表实际计算输出为 FP16，设备为 cuda:0。采集脚本通过临时 hook、融合函数包装及 forward 包装记录形状、dtype、device、输入长度和缓存长度，结束时移除观察窗口。由于上游直接调用 self.forward，不能仅靠顶层 Module forward hook 捕获此次调用。

运行入口是在项目根目录执行：

```powershell
& .\.venv\python.exe -X utf8 -u .\scripts\trace_vlm_shapes.py
```

本次 seed=42、do_sample=False、use_cache=True、max_new_tokens=2，CUDA FP16。原始 JSON 保存软件版本、模型配置、图片与脚本哈希、运行时上游 diff 及 token ID。日志中的 65.09M 沿用上游排除视觉编码器的参数统计；本次没有测量正式速度或显存，也不以“这张”评价图片问答质量。

## 四、现有注释代码的逐项记录与澄清

当前注释由学习过程中双方陆续补充，不能仅凭 Git diff 精确归属每行作者。下述记录保留当前文件中的注释原文，源码不作修改；附录提供包含这些注释的完整函数快照及行号。

| 文件 / 代码位置 | 注释覆盖内容 | 对应含义与澄清 |
| --- | --- | --- |
| eval_vlm.py / init_model | ckp 权重路径 | 当前 dense 配置加载 out/sft_vlm_768.pth；视觉编码器另有权重目录 |
| eval_vlm.py / 图片扩展名判断 | 支持 png、jpg、jpeg、bmp | 不匹配则跳过该循环体；“不然就没有 pixel_values”不表示其他路径不能产生图片张量 |
| eval_vlm.py / RGB 转换 | 三通道图片 | convert('RGB') 统一图像模式，processor 随后处理尺寸与像素数值 |
| eval_vlm.py / 字典推导式 | k、v、image2tensor、to(device) | 图片预处理返回字典，逐个迁移张量，字典本身不是张量 |
| eval_vlm.py / messages | role/content、64 个占位符 | 是本地对话模板所需消息结构，不代表向外部 OpenAI 接口发送请求；64 来自当前配置 |
| eval_vlm.py / 模板与 tokenizer | 对话编码与迁移设备 | tokenize=False 的 apply_chat_template 只生成字符串，真正编码发生在下一句 tokenizer 调用 |
| eval_vlm.py / model.generate | model 实例与生成参数 | 当前原生权重路径是 MiniMindVLM 实例；实际循环位于父类 MiniMindForCausalLM.generate |
| model_vlm.py / embed_tokens | ID 转嵌入，batch/seq/hidden | 本次 [1,93]→[1,93,768]；这时图片位置尚未替换 |
| model_vlm.py / 图片处理条件 | pixel_values 与 start_pos | 有图且没有已有位置缓存时处理图像；“序列开头”在本实现中由缓存长度判断 |
| model_vlm.py / 五维与四维分支 | 多图与单图 | 五维是 [batch,num_images,C,H,W]，增加的是图片数量轴，不是另一个 batch_size；四维为 [batch,C,H,W]，不必然 batch=1 |
| model_vlm.py / get_image_embeddings、vision_proj | 视觉编码与投影结构 | 编码器已输出张量，投影进一步变换特征，而非首次“转换为张量” |
| model_vlm.py / count_vision_proj | 替换前后示意 | 替换占位符嵌入，不新增位置；图中省略了部分模板标记 |
| model_vlm.py / presents、循环、layer 返回 | 层间隐藏状态与每层缓存 | layers 中实际是 MiniMindBlock，现有注释写 MiniMindLayer 不够准确；返回变量名为 present，收集到 presents 后作为 past_key_values 返回 |

当前相对固定提交的 model_minimind.py 没有本地 diff。生成循环虽然已经讲解，但不能记成用户在那里添加过注释。下面同时保留其实际完整实现，便于从 VLM 调用追踪到父类。

完整代码与注释行号索引见 [annotated-code.md](annotated-code.md)
