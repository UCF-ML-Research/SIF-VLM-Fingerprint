import os
import json
import argparse
from PIL import Image
import torch
from utils import QA_PAIRS, parse_float_expr, set_seed, get_device, save_rgb, load_llava, load_qwen, run_inference_llava, run_inference_qwen
from core.pla import PLA
from core.ordinary import Ordinary
from core.rna import RNA
from core.cropa import CroPA
from core.difgsm import DIFGSM
from core.proflingo import ProFLingo
from core.instruction_fingerprint import InstructionFingerprint

METHODS = ["pla", "ordinary", "rna", "cropa", "difgsm", "proflingo", "instruction_fingerprint"]


def load_model_for_proflingo(model_name, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    try:
        from transformers import LlavaForConditionalGeneration
    except ImportError:
        LlavaForConditionalGeneration = None

    lower = model_name.lower()
    if "llava" in lower:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True, device_map="auto")
        tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True).tokenizer
    elif "qwen" in lower and "vl" in lower:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
            device_map="auto", trust_remote_code=True)
        tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True).tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
            device_map="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return model, tokenizer


def run_proflingo(args, dtype_model):
    model, tokenizer = load_model_for_proflingo(args.model_name, dtype_model)
    pf = ProFLingo(model, tokenizer)
    pf.generate_all(args.out_dir, num_epoch=args.num_epoch,
                    token_nums=args.token_nums, seed=args.seed,
                    num_questions=args.num_questions)


def run_instruction_fingerprint(args, dtype_model):
    model, tokenizer = load_model_for_proflingo(args.model_name, dtype_model)
    ifp = InstructionFingerprint(model, tokenizer)
    ifp.embed(args.out_dir, num_fingerprint=args.num_fingerprint,
              num_epochs=args.if_epochs, lr=args.if_lr,
              inner_dim=args.if_inner_dim,
              batch_size=args.if_batch_size, seed=args.seed,
              mode=args.if_mode, lora_r=args.if_lora_r, lora_alpha=args.if_lora_alpha)
    ifp.verify(pairs_path=os.path.join(args.out_dir, "fingerprint_pairs.json"))


def run_image_attack(args, dtype_model):
    set_seed(args.seed)
    device = get_device(args.primary)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.model_type == "llava":
        model, processor = load_llava(args.model_name, dtype_model, device_map="auto")
    else:
        model, processor = load_qwen(args.model_name, dtype_model, device_map="auto")

    if args.method == "pla":
        attacker = PLA(model, processor, args.model_type, dtype_model, device)
    elif args.method == "ordinary":
        attacker = Ordinary(model, processor, args.model_type, dtype_model, device)
    elif args.method == "rna":
        attacker = RNA(model, processor, args.model_type, dtype_model, device)
    elif args.method == "cropa":
        attacker = CroPA(model, processor, args.model_type, dtype_model, device)
    else:
        attacker = DIFGSM(model, processor, args.model_type, dtype_model, device)

    img_files_all = sorted([
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    total = len(img_files_all)
    start = max(1, args.start)
    end = min(args.end, total) if args.end else total
    img_files = img_files_all[start-1:end]

    for gidx_offset, img_path in enumerate(img_files):
        gidx = start + gidx_offset
        pil = Image.open(img_path).convert("RGB")
        for pid, (question, target_text) in enumerate(QA_PAIRS):
            out_dir = os.path.join(args.out_dir, f"img{gidx:04d}_pair{pid}")
            os.makedirs(out_dir, exist_ok=True)

            if args.method == "pla":
                adv_rgb = attacker.attack(pil, question, target_text,
                                          args.steps, args.eps, args.alpha, args.beta, args.clip_th)
            elif args.method == "ordinary":
                adv_rgb = attacker.attack(pil, question, target_text,
                                          args.steps, args.eps, args.alpha)
            elif args.method == "rna":
                adv_rgb = attacker.attack(pil, question, target_text,
                                          args.steps, args.eps, args.alpha, args.lam)
            elif args.method == "cropa":
                adv_rgb = attacker.attack(pil, question, target_text,
                                          args.steps, args.eps, args.alpha, args.alpha2, args.cropa_end)
            else:
                adv_rgb = attacker.attack(pil, question, target_text,
                                          args.steps, args.eps, args.alpha, args.momentum,
                                          args.di_prob, args.di_resize_range)

            adv_path = os.path.join(out_dir, "adv_pixel_vis.png")
            save_rgb(adv_rgb, adv_path)
            torch.save(adv_rgb.detach().cpu(), os.path.join(out_dir, "adv_pixel_vis.pt"))

            if args.model_type == "llava":
                model_output = run_inference_llava(model, processor, adv_path, question, dtype_model, device)
            else:
                model_output = run_inference_qwen(model, processor, adv_path, question, dtype_model, device)

            record = {
                "original_image": img_path, "adv_image": adv_path,
                "question": question, "target_text": target_text,
                "model_output": model_output, "method": args.method,
                "model_name": args.model_name, "index": gidx,
            }
            with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as jf:
                json.dump(record, jf, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", type=str, choices=METHODS, required=True)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], required=True)
    # Image attack args
    p.add_argument("--model_type", type=str, choices=["llava", "qwen"], default="llava")
    p.add_argument("--input_dir", type=str, default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--eps", type=parse_float_expr, default=16/255)
    p.add_argument("--alpha", type=parse_float_expr, default=1/255)
    p.add_argument("--beta", type=float, default=1e-4)
    p.add_argument("--clip_th", type=float, default=5e-3)
    p.add_argument("--lam", type=float, default=1e-4)
    p.add_argument("--alpha2", type=float, default=0.01)
    p.add_argument("--cropa_end", type=int, default=300)
    p.add_argument("--momentum", type=float, default=1.0)
    p.add_argument("--di_prob", type=float, default=0.5)
    p.add_argument("--di_resize_range", type=int, default=30)
    p.add_argument("--primary", type=int, default=0)
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=None)
    # ProFLingo args
    p.add_argument("--num_epoch", type=int, default=64)
    p.add_argument("--token_nums", type=int, default=32)
    p.add_argument("--num_questions", type=int, default=50)
    # Instruction Fingerprint args
    p.add_argument("--num_fingerprint", type=int, default=10)
    p.add_argument("--if_epochs", type=int, default=20)
    p.add_argument("--if_lr", type=float, default=2e-4)
    p.add_argument("--if_inner_dim", type=int, default=16)
    p.add_argument("--if_batch_size", type=int, default=1)
    p.add_argument("--if_mode", choices=["adapter", "lora"], default="lora")
    p.add_argument("--if_lora_r", type=int, default=16)
    p.add_argument("--if_lora_alpha", type=int, default=32)
    args = p.parse_args()

    dtype_model = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.method == "proflingo":
        run_proflingo(args, dtype_model)
    elif args.method == "instruction_fingerprint":
        run_instruction_fingerprint(args, dtype_model)
    else:
        run_image_attack(args, dtype_model)


if __name__ == "__main__":
    main()
