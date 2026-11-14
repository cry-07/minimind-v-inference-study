# 第二阶段：当前注释代码与完整函数快照

来源：MiniMind-V，Apache-2.0；上游版权与许可见 [第三方说明](../THIRD_PARTY_NOTICES.md)。以下摘录含本地教学注释，原始推理实现来自上游。

快照日期：2026-11-14。以下直接提取当前源文件，保留原注释、分支和语句。原有注释的不准确之处见 [inference-path.md](inference-path.md) 第四部分；本文件不修改源代码，也不把所有本地注释都归属某一位作者。行号对应本次快照，后续编辑可能变化。

## vendor/minimind-v/eval_vlm.py

文件 SHA256：`6869d2330a6a0b724c9ca1cdebd5020d5d284da34d49af1eb2552cb56f9fabdd`

### 中文注释定位索引

| 行号 | 当前注释原文 |
| --- | --- |
| 23 | # 加载模型权重时，这里的 ckp 就是模型权重文件：./out/sft_vlm_768.pth |
| 55 | # 自动测试image_dir中的所有图像 |
| 59 | # 如果图片文件名以 .png、.jpg、.jpeg 或 .bmp 结尾，则进行处理 |
| 60 | # 不然 就没有 pixel_values |
| 64 | # 读取图片并统一为 RGB 三通道，以匹配视觉编码器输入 |
| 68 | # 将图片预处理为张量，并移动到模型所在设备（如 GPU）。 |
| 70 | # k 是键, v 是值(张量)，v.to(args.device) 将张量移动到指定设备（如 GPU 或 CPU）。 |
| 71 | # 移到模型所在设备 |
| 73 | # 调用 MiniMindVLM.image2tensor 方法，将图像和预处理器传入，返回一个字典，包含图像张量 v |
| 77 | # messages 是对话消息列表，遵循 OpenAI 接口，有以下格式： |
| 82 | # <image> 是占位符，表示图像位置 |
| 83 | # 一共拼接 64 个占位符 |
| 90 | # 将对话消息 messages 编码为 token ID ,并将其转换为 PyTorch 张量，移动到模型所在设备 |
| 99 | # 这里的 Model 是 MiniMindVLM 或 AutoModelForCausalLM 的实例 |
| 100 | # generate 方法用于生成文本，传入输入的 token ID、注意力掩码、最大生成长度、采样参数等 |

### init_model

源文件第 13—33 行

```python
def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from, trust_remote_code=True)
    if os.path.basename(os.path.normpath(args.load_from)) == 'model':
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniMindVLM(
            VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)),
            vision_model_path="./model/siglip2-base-p32-256-ve"
        )

        # 加载模型权重时，这里的 ckp 就是模型权重文件：./out/sft_vlm_768.pth
        state_dict = torch.load(ckp, map_location=args.device)
        
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        model.vision_encoder, model.processor = MiniMindVLM.get_vision_model("./model/siglip2-base-p32-256-ve")
    get_model_params(model, model.config)
    model = model.eval()
    if "cuda" in args.device: model = model.half()
    return model.to(args.device), tokenizer, model.processor
```

### main

源文件第 36—108 行

```python
def main():
    parser = argparse.ArgumentParser(description="MiniMind-V Chat")
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='sft_vlm', type=str, help="权重名称前缀（pretrain_vlm, sft_vlm）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="最大生成长度")
    parser.add_argument('--temperature', default=0.7, type=float, help="生成温度，控制随机性（0-1，越大越随机）")
    parser.add_argument('--top_p', default=0.85, type=float, help="nucleus采样阈值（0-1）")
    parser.add_argument('--image_dir', default='./dataset/eval_images/', type=str, help="测试图像目录")
    parser.add_argument('--show_speed', default=1, type=int, help="显示decode速度（tokens/s）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否开启自适应思考（0=否，1=是）")
    args = parser.parse_args()
    
    model, tokenizer, preprocess = init_model(args)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # 自动测试image_dir中的所有图像
    prompt = "<image>\n请描述这张图中的主要物体和场景。"
    # prompt = "<image>\nPlease illustrate the image through your words."
    for image_file in sorted(os.listdir(args.image_dir)):
        # 如果图片文件名以 .png、.jpg、.jpeg 或 .bmp 结尾，则进行处理
        # 不然 就没有 pixel_values
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            setup_seed(random.randint(1, 31415926))

            # 读取图片并统一为 RGB 三通道，以匹配视觉编码器输入
            image_path = os.path.join(args.image_dir, image_file)
            image = Image.open(image_path).convert('RGB')

            # 将图片预处理为张量，并移动到模型所在设备（如 GPU）。
            pixel_values = {
                # k 是键, v 是值(张量)，v.to(args.device) 将张量移动到指定设备（如 GPU 或 CPU）。
                # 移到模型所在设备
                k: v.to(args.device)
                # 调用 MiniMindVLM.image2tensor 方法，将图像和预处理器传入，返回一个字典，包含图像张量 v
                for k, v in MiniMindVLM.image2tensor(image, preprocess).items()
            }
            
            # messages 是对话消息列表，遵循 OpenAI 接口，有以下格式：
            # messages = [
            #    {"role": "user", "content": "..."}
            #]
            messages = [
                # <image> 是占位符，表示图像位置
                # 一共拼接 64 个占位符
                {"role": "user", "content": 
                 prompt.replace('<image>', 
                                model.config.image_special_token * model.config.image_token_len)
                }
            ]

            # 将对话消息 messages 编码为 token ID ,并将其转换为 PyTorch 张量，移动到模型所在设备
            inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
            inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)
            
            print(f'[图像]: {image_file}')
            print(f"💬: {repr(prompt)}")
            print('🤖: ', end='')
            st = time.time()

            # 这里的 Model 是 MiniMindVLM 或 AutoModelForCausalLM 的实例
            # generate 方法用于生成文本，传入输入的 token ID、注意力掩码、最大生成长度、采样参数等
            generated_ids = model.generate(
                inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                top_p=args.top_p, temperature=args.temperature, pixel_values=pixel_values
            )
            gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
            print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')
```

## vendor/minimind-v/model/model_vlm.py

文件 SHA256：`40e3c456506e158c3ca1e51619703281b0145476159ff07bdab5a67fdfc63c60`

### 中文注释定位索引

| 行号 | 当前注释原文 |
| --- | --- |
| 35 | # 继承自语言模型 |
| 56 | # 冻结 vision_encoder 的所有参数 |
| 112 | # 词元嵌入，把输入的 token ID 转换为嵌入向量，形状 (batch, seq_len, hidden_size(嵌入向量的维数)) |
| 115 | # 则将图像张量投影到语言模型的隐藏空间，并将其插入到隐藏状态中 |
| 116 | # pixel_values is not None 表示输入中包含图像张量 |
| 117 | # start_pos == 0 表示这是序列的开头 |
| 121 | # 如果 sample_val 的维度是 5，说明这是多个输入(多了一个 batch_size) |
| 125 | # 代表单个图片，有四个维度 |
| 127 | # get_image_embeddings(...)：用 SigLIP 视觉编码器提取图片特征 |
| 128 | # vision_proj(...)：经过 LayerNorm → Linear → GELU → Linear，把视觉特征转换为张量 |
| 138 | # 替换前：[对话前缀嵌入][64 个占位符嵌入][问题嵌入] |
| 139 | # 替换后：[对话前缀嵌入][64 个视觉向量  ][问题嵌入] |
| 156 | # presents 是一个列表，用于存储每一层的 past_key_value |
| 157 | # hidden_states 是当前层的输入，经过每一层的计算后，hidden_states 会被更新为下一层的输入 |
| 160 | # layer() 方法是 MiniMindLayer 的前向传播函数，返回当前层的输出 hidden_states 和 past_key_value |

### MMVisionProjector

源文件第 23—33 行

```python
class MMVisionProjector(nn.Module):
    def __init__(self, in_dim, out_dim, source_tokens=64, target_tokens=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.mlp(x)
```

### MiniMindVLM.image2tensor

源文件第 61—65 行

```python
    @staticmethod
    def image2tensor(image, processor):
        if image.mode in ['RGBA', 'LA']: image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        return inputs
```

### MiniMindVLM.get_image_embeddings

源文件第 67—73 行

```python
    @staticmethod
    def get_image_embeddings(image_inputs, vision_model):
        if hasattr(image_inputs, 'keys'):
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()}
        with torch.no_grad():
            outputs = vision_model(**image_inputs)
        return outputs.last_hidden_state
```

### MiniMindVLM.count_vision_proj

源文件第 75—96 行

```python
    @torch.compiler.disable
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        if vision_tensors is None or not self.config.image_ids:
            return h
        marker, vf = self.config.image_ids[0], vision_tensors
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)
        out = []
        for b in range(h.size(0)):
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
                        k += 1
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)
```

### MiniMindVLM.forward

源文件第 98—184 行

```python
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                labels: Optional[torch.Tensor] = None,
                pixel_values: Optional[torch.FloatTensor] = None,
                **args):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.model.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 词元嵌入，把输入的 token ID 转换为嵌入向量，形状 (batch, seq_len, hidden_size(嵌入向量的维数))
        hidden_states = self.model.dropout(self.model.embed_tokens(input_ids))

        # 则将图像张量投影到语言模型的隐藏空间，并将其插入到隐藏状态中
        # pixel_values is not None 表示输入中包含图像张量
        # start_pos == 0 表示这是序列的开头
        if pixel_values is not None and start_pos == 0:
            if hasattr(pixel_values, 'keys'):
                sample_val = next(iter(pixel_values.values()))
                # 如果 sample_val 的维度是 5，说明这是多个输入(多了一个 batch_size)
                if sample_val.ndim == 5:
                    bs, num = sample_val.shape[:2]
                    vision_tensors = self.vision_proj(MiniMindVLM.get_image_embeddings({k: v.flatten(0, 1) for k, v in pixel_values.items()}, self.vision_encoder)).view(bs, num, self.config.image_token_len, -1)
                # 代表单个图片，有四个维度
                else:
                    # get_image_embeddings(...)：用 SigLIP 视觉编码器提取图片特征
                    # vision_proj(...)：经过 LayerNorm → Linear → GELU → Linear，把视觉特征转换为张量
                    vision_tensors = self.vision_proj(
                        MiniMindVLM.get_image_embeddings(pixel_values, self.vision_encoder)
                    )
            else:
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                bs, num, c, im_h, im_w = pixel_values.shape
                vision_tensors = torch.stack([self.vision_proj(MiniMindVLM.get_image_embeddings(pixel_values[:, i, :, :, :], self.vision_encoder)) for i in range(num)], dim=1)
            
            # 替换前：[对话前缀嵌入][64 个占位符嵌入][问题嵌入]
            # 替换后：[对话前缀嵌入][64 个视觉向量  ][问题嵌入]
            hidden_states = self.count_vision_proj(
                tokens=input_ids, 
                h=hidden_states, 
                vision_tensors=vision_tensors, 
                seqlen=input_ids.shape[1]
            )

        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.model.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.model.freqs_cos, self.model.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (
            self.model.freqs_cos[start_pos:start_pos + seq_length],
            self.model.freqs_sin[start_pos:start_pos + seq_length]
        )

        # presents 是一个列表，用于存储每一层的 past_key_value
        # hidden_states 是当前层的输入，经过每一层的计算后，hidden_states 会被更新为下一层的输入
        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.model.layers, past_key_values)):
            # layer() 方法是 MiniMindLayer 的前向传播函数，返回当前层的输出 hidden_states 和 past_key_value
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.model.norm(hidden_states)

        aux_loss = sum([l.mlp.aux_loss for l in self.model.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        aux_loss = aux_loss + sum(p.sum() for p in self.vision_proj.parameters()) * 0  # dummy gradient for DDP
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        output = MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=presents, hidden_states=hidden_states)
        return output
```

### MiniMindVLM.generate

源文件第 186—193 行

```python
    def generate(self, *args, num_return_sequences=1, **kwargs):
        if num_return_sequences > 1 and 'pixel_values' in kwargs:
            pv = kwargs['pixel_values']
            if hasattr(pv, 'keys'):
                kwargs['pixel_values'] = {k: v.repeat(num_return_sequences, *([1] * (v.ndim - 1))) for k, v in pv.items()}
            else:
                kwargs['pixel_values'] = pv.repeat(num_return_sequences, *([1] * (pv.ndim - 1)))
        return super().generate(*args, num_return_sequences=num_return_sequences, **kwargs)
```

## vendor/minimind-v/model/model_minimind.py

文件 SHA256：`0eeb8e1b4a0ac1194ea83013d62271722c05f6bfd230e7ddafc1ee0c2a9df094`

### MiniMindForCausalLM.generate

源文件第 261—293 行

```python
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
```

## 采集脚本的解码位置

` scripts/trace_vlm_shapes.py ` 在生成结束后切出新增 ID，再通过 `tokenizer.decode(new_ids, skip_special_tokens=True)` 得到文字。原生 eval_vlm.py 则通过 TextStreamer 显示生成文本，不能把采集脚本的 decode 语句误认为位于上游 generate 内部。
