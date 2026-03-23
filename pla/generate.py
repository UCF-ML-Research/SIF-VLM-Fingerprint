import os
import json
import argparse
from PIL import Image
import torch
from utils import QA_PAIRS, parse_float_expr, set_seed, get_device, save_rgb, load_llava, load_qwen, run_inference_llava, run_inference_qwen
from pla import PLA

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--model_type", type=str, choices=["llava", "qwen"], required=True)
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--eps", type=parse_float_expr, required=True)
    p.add_argument("--alpha", type=parse_float_expr, required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--clip_th", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], required=True)
    p.add_argument("--primary", type=int, default=0)
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=None)
    args = p.parse_args()
    if args.dtype == "bf16":
        dtype_model = torch.bfloat16
    elif args.dtype == "fp16":
        dtype_model = torch.float16
    else:
        dtype_model = torch.float32
    set_seed(args.seed)
    device = get_device(args.primary)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.model_type == "llava":
        model, processor = load_llava(args.model_name, dtype_model, device_map="auto")
    else:
        model, processor = load_qwen(args.model_name, dtype_model, device_map="auto")
    pla = PLA(model, processor, args.model_type, dtype_model, device)
    img_files_all = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    total = len(img_files_all)
    start = max(1, args.start)
    if args.end is not None:
        end = min(args.end, total)
    else:
        end = total
    if start > end:
        raise ValueError(f"No images to process: start={start}, end={end}, total={total}")
    img_files = img_files_all[start-1:end]
    for gidx_offset, img_path in enumerate(img_files):
        gidx = start + gidx_offset
        pil = Image.open(img_path).convert("RGB")
        for pid, (question, target_text) in enumerate(QA_PAIRS):
            out_dir = os.path.join(args.out_dir, f"img{gidx:04d}_pair{pid}")
            os.makedirs(out_dir, exist_ok=True)
            adv_rgb = pla.attack(
                pil, question, target_text,
                args.steps, args.eps, args.alpha, args.beta, args.clip_th
            )
            adv_image_path_png = os.path.join(out_dir, "adv_pixel_vis.png")
            save_rgb(adv_rgb, adv_image_path_png)
            torch.save(adv_rgb.detach().cpu(), os.path.join(out_dir, "adv_pixel_vis.pt"))
            if args.model_type == "llava":
                model_output = run_inference_llava(model, processor, adv_image_path_png, question, dtype_model, device)
            else:
                model_output = run_inference_qwen(model, processor, adv_image_path_png, question, dtype_model, device)
            record = {
                "original_image": img_path,
                "adv_image": adv_image_path_png,
                "question": question,
                "target_text": target_text,
                "model_output": model_output,
                "model_name": args.model_name,
                "model_type": args.model_type,
                "index": gidx
            }
            with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as jf:
                json.dump(record, jf, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
