"""M2 教学观察入口：只观察形状，不修改模型运算或测量性能。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'vendor' / 'minimind-v'


def main():
    # 1. 先解析参数；--help 不导入 torch，也不加载模型。
    parser = argparse.ArgumentParser(description='M2：采集单图推理关键张量形状')
    parser.add_argument('--image', type=Path, default=UPSTREAM / 'dataset/eval_images/image-01-golden-dog-balloons.jpg')
    args = parser.parse_args()
    image_path = args.image.resolve()
    run_name = datetime.now().strftime('m2-shapes-%Y%m%d-%H%M%S-') + uuid4().hex[:8]
    output_path = ROOT / 'results/raw' / (run_name + '.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {'status': 'started', 'events': [], 'image': str(image_path)}
    handles = []
    original_cwd = Path.cwd()
    model = None
    original_merge = None
    had_instance_merge = False
    saved_instance_merge = None

    def observe(name, value):
        # 只读取元信息，不保存大张量或更改数值。
        event = {'name': name, 'shape': list(value.shape),
                 'dtype': str(value.dtype), 'device': str(value.device)}
        record['events'].append(event)
        print(f"  {name}: shape={event['shape']}, {event['dtype']}, {event['device']}")

    try:
        # 2. 固定本地资源与生成设置，沿用上游的相对资源路径。
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        sys.path.insert(0, str(UPSTREAM))
        os.chdir(UPSTREAM)
        import torch
        import transformers
        from PIL import Image
        from eval_vlm import init_model
        from model.model_vlm import MiniMindVLM
        from trainer.trainer_utils import setup_seed

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA 不可用；停止，不静默改用 CPU。')
        record.update({
            'upstream_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'upstream_diff': subprocess.check_output(['git', 'diff', '--', '*.py'], text=True, encoding='utf-8'),
            'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'python': sys.version, 'torch': torch.__version__,
            'transformers': transformers.__version__, 'cuda_build': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0),
            'generation': {'seed': 42, 'do_sample': False, 'max_new_tokens': 2,
                           'use_cache': True, 'temperature': 1.0, 'top_k': 0,
                           'top_p': 1.0, 'repetition_penalty': 1.0},
        })
        setup_seed(42)
        print('[1] 加载与 M1 相同的本地模型（CUDA FP16）')
        config = argparse.Namespace(load_from='model', save_dir='out', weight='sft_vlm',
                                    hidden_size=768, num_hidden_layers=8, use_moe=0, device='cuda')
        model, tokenizer, processor = init_model(config)
        if model.vision_encoder is None or processor is None:
            raise RuntimeError('本地视觉编码器或预处理器未加载。')
        record['model_config'] = model.config.to_dict()

        # 3. 图片与文本分别准备，与 eval_vlm.py 的单图路径一致。
        print('[2] 准备图片张量、对话文本和 token ID')
        with Image.open(image_path) as source:
            record['original_image_size'] = list(source.size)
            image = source.convert('RGB')
        pixels = {k: v.to('cuda') for k, v in MiniMindVLM.image2tensor(image, processor).items()}
        for key, value in pixels.items():
            observe('image.' + key, value)
        prompt = '<image>\n请描述这张图中的主要物体和场景。'
        messages = [{'role': 'user', 'content': prompt.replace('<image>', model.config.image_special_token * model.config.image_token_len)}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=False)
        inputs = tokenizer(text, return_tensors='pt', truncation=True).to('cuda')
        record.update({'prompt': prompt, 'formatted_text': text,
                       'input_ids': inputs.input_ids.tolist(),
                       'image_marker_id': model.config.image_ids[0],
                       'image_marker_count': int((inputs.input_ids == model.config.image_ids[0]).sum().item())})
        observe('text.input_ids', inputs.input_ids)
        observe('text.attention_mask', inputs.attention_mask)
        print('  图片占位符数量:', record['image_marker_count'])

        # 4. hook 像观察窗口：模块照常运算，回调只打印输入/输出形状。
        print('[3] 安装临时观察窗口')
        handles.append(model.model.embed_tokens.register_forward_hook(lambda m, a, o: observe('text.embeddings', o)))
        handles.append(model.vision_encoder.register_forward_hook(lambda m, a, o: observe('vision.last_hidden_state', o.last_hidden_state)))
        handles.append(model.vision_proj.register_forward_hook(lambda m, a, o: observe('vision.projected', o)))
        handles.append(model.model.layers[0].register_forward_pre_hook(lambda m, a: observe('transformer.layer0.input', a[0])))
        handles.append(model.lm_head.register_forward_hook(lambda m, a, o: observe('language.logits', o)))

        # 融合函数不是 nn.Module，临时包装它来观察替换前后，再原样返回结果。
        had_instance_merge = 'count_vision_proj' in model.__dict__
        saved_instance_merge = model.__dict__.get('count_vision_proj')
        original_merge = model.count_vision_proj

        def observe_merge(*a, **kw):
            observe('merge.before', kw['h'])
            result = original_merge(*a, **kw)
            observe('merge.after', result)
            return result

        model.count_vision_proj = observe_merge

        def before_forward(module, positional, keywords):
            ids = keywords.get('input_ids', positional[0] if positional else None)
            cache = keywords.get('past_key_values', positional[2] if len(positional) > 2 else None)
            past_length = cache[0][0].shape[1] if cache and cache[0] is not None else 0
            record['events'].append({'name': 'forward', 'input_length': ids.shape[1], 'past_length': past_length})
            print(f'  forward: 本次输入长度={ids.shape[1]}, 已缓存长度={past_length}')

        # 上游 generate 直接调用 self.forward，因此包装 forward 本身。
        original_forward = model.forward

        def traced_forward(*a, **kw):
            before_forward(model, a, kw)
            return original_forward(*a, **kw)

        model.forward = traced_forward
        try:
            # 5. 两步贪心生成，仅用于观察 prefill 与 decode。
            print('[4] 最多生成 2 个 token，记录真实路径')
            with torch.inference_mode():
                generated = model.generate(
                    inputs=inputs.input_ids, attention_mask=inputs.attention_mask,
                    pixel_values=pixels, max_new_tokens=2, do_sample=False,
                    use_cache=True, temperature=1.0, top_k=0, top_p=1.0,
                    repetition_penalty=1.0, pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            new_ids = generated[0, inputs.input_ids.shape[1]:].tolist()
            record.update({'generated_token_ids': new_ids,
                           'generated_text': tokenizer.decode(new_ids, skip_special_tokens=True),
                           'stop_reason': 'eos' if new_ids and new_ids[-1] == tokenizer.eos_token_id else 'length_limit',
                           'status': 'completed'})
        finally:
            del model.forward
    except Exception:
        record['status'] = 'failed'
        record['error'] = traceback.format_exc()
        raise
    finally:
        # 6. 无论成功失败都移除观察窗口，保存独立记录，不覆盖已有结果。
        for handle in handles:
            handle.remove()
        if original_merge is not None:
            if had_instance_merge:
                model.count_vision_proj = saved_instance_merge
            else:
                del model.count_vision_proj
        os.chdir(original_cwd)
        with output_path.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
        print('[5] 记录已保存:', output_path)


if __name__ == '__main__':
    main()
