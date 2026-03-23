import os
import json
import argparse
import random
import ast
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image, resize as tv_resize
from torchvision.transforms import InterpolationMode
from transformers import AutoProcessor, AutoModelForImageTextToText, LogitsProcessorList
from watermarks.kgw.watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

def save_rgb(bchw, path):
    t = bchw.detach().cpu()
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    t = t.clamp(0, 1)
    to_pil_image(t).save(path)

def to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        if obj.dim() == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj

class DiffQwen2VLFast(nn.Module):
    def __init__(self, processor: AutoProcessor, pil_image: Image.Image, device=None):
        super().__init__()
        self.device_ = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ip = processor.image_processor
        self.patch = int(getattr(self.ip, "patch_size", 14))
        self.temporal_patch = int(getattr(self.ip, "temporal_patch_size", 2))
        self.merge_size = int(getattr(self.ip, "merge_size", 1))
        self.interp = InterpolationMode.BICUBIC
        mean = torch.tensor(getattr(self.ip, "image_mean")).view(1, 3, 1, 1)
        std = torch.tensor(getattr(self.ip, "image_std")).view(1, 3, 1, 1)
        self.register_buffer("mean", mean.to(self.device_, dtype=torch.float32))
        self.register_buffer("std", std.to(self.device_, dtype=torch.float32))
        with torch.no_grad():
            probe = processor(images=[pil_image], text="probe", return_tensors="pt")
        pv = probe["pixel_values"]
        if pv.dim() == 2:
            pv = pv.unsqueeze(0)
        self.pv_ref = pv.to(self.device_, dtype=torch.float32)
        t, gh, gw = map(int, probe["image_grid_thw"][0].tolist())
        self.H = gh * self.patch
        self.W = gw * self.patch
        self.grid_thw = torch.tensor([[t, gh, gw]], device=self.device_, dtype=torch.long)

    @property
    def device(self):
        return self.device_

    def _resize_norm(self, bchw: torch.Tensor) -> torch.Tensor:
        x = bchw.clamp(0, 1).to(torch.float32)
        x = tv_resize(x, [self.H, self.W], interpolation=self.interp, antialias=True)
        x = (x - self.mean) / self.std
        return x

    def forward(self, bchw: torch.Tensor):
        x = bchw.to(self.device, dtype=torch.float32)
        x = self._resize_norm(x)
        patches = x.unsqueeze(1)
        if patches.shape[1] % self.temporal_patch != 0:
            rep = self.temporal_patch - (patches.shape[1] % self.temporal_patch)
            patches = torch.cat([patches, patches[:, -1:].repeat(1, rep, 1, 1, 1)], dim=1)
        B, grid_t_mul, C, H, W = patches.shape
        grid_t = grid_t_mul // self.temporal_patch
        grid_h = H // self.patch
        grid_w = W // self.patch
        patches = patches.view(
            B,
            grid_t,
            self.temporal_patch,
            C,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        T = grid_t * grid_h * grid_w
        D = C * self.temporal_patch * self.patch * self.patch
        tokens = patches.reshape(B, T, D).contiguous()
        return tokens, self.grid_thw

def parse_float_expr(x):
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    try:
        tree = ast.parse(s, mode="eval")
        return float(_eval_ast(tree))
    except Exception:
        return float(s)

_ALLOWED_BINOP = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_ALLOWED_UNARYOP = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}

def _eval_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError
    if isinstance(node, ast.Num):
        return float(node.n)
    if isinstance(node, ast.BinOp):
        return _ALLOWED_BINOP[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp):
        return _ALLOWED_UNARYOP[type(node.op)](_eval_ast(node.operand))
    raise ValueError

def qwen_messages(image, prompt):
    return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

@torch.no_grad()
def gen_watermarked_target(model, processor, tokenizer, pil_image, prompt, dtype, device, gamma, delta, seeding_scheme, z_threshold, min_tokens=84, max_retries=10):
    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(range(len(tokenizer))),
        gamma=gamma,
        delta=delta,
        seeding_scheme=seeding_scheme,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    pre = DiffQwen2VLFast(processor, pil_image, device=device)
    messages = qwen_messages(pil_image, prompt)
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    tok, grid = pre(ToTensor()(pil_image).unsqueeze(0).to(device))
    inputs["pixel_values"] = tok.to(dtype)
    inputs["image_grid_thw"] = grid
    if getattr(model.generation_config, "pad_token_id", None) is None:
        pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
        if pad_id is not None:
            model.generation_config.pad_token_id = pad_id
    attempt = 0
    final_text = None
    gen_ids_only = None
    use_amp = torch.cuda.is_available() and dtype in (torch.float16, torch.bfloat16)
    while attempt < max_retries:
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype if use_amp else None):
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                logits_processor=LogitsProcessorList([watermark_processor]),
                return_dict_in_generate=True,
                output_scores=False,
            )
        seq = out.sequences
        prefix_len = inputs["input_ids"].shape[1]
        gen_only = seq[:, prefix_len:]
        if gen_only.shape[1] >= min_tokens:
            gen_text = processor.batch_decode(gen_only, skip_special_tokens=True)[0].strip()
            final_text = gen_text
            gen_ids_only = gen_only[0].tolist()
            break
        attempt += 1
    if final_text is None:
        prefix_len = inputs["input_ids"].shape[1]
        gen_text = processor.batch_decode(out.sequences[:, prefix_len:], skip_special_tokens=True)[0].strip()
        final_text = gen_text
        gen_ids_only = out.sequences[0, prefix_len:].tolist()
    return final_text, gen_ids_only

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--eps", type=parse_float_expr, required=True)
    p.add_argument("--alpha", type=parse_float_expr, required=True)
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--delta", type=float, required=True)
    p.add_argument("--seeding_scheme", type=str, required=True)
    p.add_argument("--z_threshold", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], required=True)
    p.add_argument("--primary", type=int, default=0)
    p.add_argument("--min_gen_tokens", type=int, default=84)
    args = p.parse_args()

    if args.dtype == "bf16":
        dtype_model = torch.bfloat16
    elif args.dtype == "fp16":
        dtype_model = torch.float16
    else:
        dtype_model = torch.float32

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device_primary = torch.device(f"cuda:{args.primary}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        torch_dtype=dtype_model,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer
    model.eval()

    for p_param in model.parameters():
        p_param.requires_grad_(False)

    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except:
            pass

    files = sorted([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    img_files = files[args.start - 1:args.end]
    use_amp = torch.cuda.is_available() and dtype_model in (torch.float16, torch.bfloat16)

    for idx, img_path in enumerate(img_files, start=args.start):
        image = Image.open(img_path).convert("RGB")
        target_text, teacher_ids = gen_watermarked_target(
            model, processor, tokenizer, image, args.prompt, dtype_model, device_primary,
            args.gamma, args.delta, args.seeding_scheme, args.z_threshold, min_tokens=args.min_gen_tokens
        )
        rgb_ref_cpu = ToTensor()(image).unsqueeze(0)
        out_dir = os.path.join(args.out_dir, f"img{idx:04d}")
        os.makedirs(out_dir, exist_ok=True)
        rgb_ref = rgb_ref_cpu.clone().to(device_primary)
        rgb = rgb_ref.clone().detach().requires_grad_(True)

        pre = DiffQwen2VLFast(processor, image, device=device_primary)
        messages_full = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": args.prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
        ]
        full_inputs = processor.apply_chat_template(messages_full, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt")
        full_inputs = {k: v.to(device_primary) for k, v in full_inputs.items()}
        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs["attention_mask"]
        messages_prompt = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": args.prompt}]}]
        prompt_inputs = processor.apply_chat_template(messages_prompt, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100

        iterator = trange(args.steps, desc=f"img{idx:04d}_PGD", ncols=100)
        for step in iterator:
            if rgb.grad is not None:
                rgb.grad = None

            tok, grid = pre(rgb)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype_model if use_amp else None):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=tok.to(dtype_model),
                    image_grid_thw=grid,
                    labels=labels
                )
                loss = outputs.loss

            if not torch.isfinite(loss):
                break

            rgb_grad, = torch.autograd.grad(loss, rgb, retain_graph=False, create_graph=False)

            with torch.no_grad():
                rgb.add_(-args.alpha * rgb_grad.sign())
                rgb.copy_(torch.max(torch.min(rgb, rgb_ref + args.eps), rgb_ref - args.eps))
                rgb.clamp_(0.0, 1.0)

            if (step % 10) == 0:
                iterator.set_postfix_str(f"loss={float(loss.item()):.6f}")

        adv_image_path_png = os.path.join(out_dir, "adv_pixel_vis.png")
        save_rgb(rgb, adv_image_path_png)
        torch.save(rgb.detach().cpu(), os.path.join(out_dir, "adv_pixel_vis.pt"))

        adv_tok, adv_grid = pre(rgb.detach())
        gen_inputs = {
            "input_ids": prompt_inputs["input_ids"].to(device_primary),
            "attention_mask": prompt_inputs["attention_mask"].to(device_primary),
            "pixel_values": adv_tok.to(dtype_model),
            "image_grid_thw": adv_grid,
        }
        with torch.no_grad():
            gen_out = model.generate(
                **gen_inputs,
                max_new_tokens=512,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=False,
            )
        seq_adv = gen_out.sequences
        adv_gen_ids = seq_adv[0, prompt_len:]
        adv_text = processor.batch_decode(adv_gen_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()

        detector = WatermarkDetector(
            vocab=list(range(len(tokenizer))),
            gamma=args.gamma,
            delta=args.delta,
            seeding_scheme=args.seeding_scheme,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tokenizer=tokenizer,
            z_threshold=args.z_threshold,
        )
        try:
            detect_result = detector.detect(adv_text)
        except Exception as e:
            detect_result = {"error": str(e)}

        rec = {
            "original_image": img_path,
            "adv_image": adv_image_path_png,
            "prompt": args.prompt,
            "target_text": target_text,
            "teacher_token_count": len(teacher_ids) if teacher_ids is not None else None,
            "adv_text": adv_text,
            "adv_token_count": len(adv_gen_ids) if adv_gen_ids is not None else None,
            "watermark_detection": detect_result
        }
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as jf:
            json.dump(to_serializable(rec), jf, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
