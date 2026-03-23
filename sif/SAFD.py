#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, argparse, random, numpy as np, torch, torch.nn.functional as F
from tqdm import trange
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, LlavaForConditionalGeneration, LogitsProcessorList
from watermarks.kgw.watermark_processor import WatermarkLogitsProcessor, WatermarkDetector


def save_rgb(bchw, path):
    t = bchw.detach().cpu()
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    t = t.clamp(0, 1)
    to_pil_image(t).save(path)


def to_python(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist() if x.numel() > 1 else float(x.detach().cpu().item())
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: to_python(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_python(v) for v in x]
    return x


class DiffLLaVAPreprocess(torch.nn.Module):
    def __init__(self, processor, device):
        super().__init__()
        ip = processor.image_processor
        self.image_size = int(ip.crop_size["height"])
        mean = torch.tensor(ip.image_mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(ip.image_std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, image_tensor):
        x = image_tensor.to(torch.float32)
        x = torch.nn.functional.interpolate(
            x, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False
        )
        return (x - self.mean) / self.std


def chat_prefix(processor, image, prompt):
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def chat_full(processor, image, prompt, assistant_text):
    msgs = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


@torch.no_grad()
def gen_teacher_targets(
    model,
    processor,
    tokenizer,
    image,
    prompt,
    dtype,
    device,
    gamma,
    delta,
    seeding_scheme,
    z_threshold,
    min_tokens=80,
    max_new_tokens=512,
    max_retries=10,
):
    detector = WatermarkDetector(
        vocab=list(range(len(tokenizer))),
        gamma=gamma,
        delta=delta,
        seeding_scheme=seeding_scheme,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tokenizer=tokenizer,
        z_threshold=z_threshold,
    )
    wm_proc = WatermarkLogitsProcessor(
        vocab=list(range(len(tokenizer))),
        gamma=gamma,
        delta=delta,
        seeding_scheme=seeding_scheme,
        device=device,
    )
    text = chat_prefix(processor, image, prompt)
    inputs = processor(text=text, images=[image], return_tensors="pt").to(device)

    # ensure pad_token_id
    if getattr(model.generation_config, "pad_token_id", None) is None:
        tok = getattr(processor, "tokenizer", None)
        pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
        if pad_id is not None:
            model.generation_config.pad_token_id = pad_id

    attempt = 0
    final_text, gen_ids_only, q_list = None, None, None
    while attempt < max_retries:
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=dtype):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                logits_processor=LogitsProcessorList([wm_proc]),
                return_dict_in_generate=True,
                output_scores=True,
            )
        seq = out.sequences
        scores = out.scores
        pref_len = inputs["input_ids"].shape[1]
        gen_only = seq[:, pref_len:]
        if gen_only.shape[1] >= min_tokens and len(scores) >= min_tokens:
            txt = processor.batch_decode(gen_only, skip_special_tokens=True)[0].strip()
            final_text = txt
            gen_ids_only = gen_only[0].tolist()
            q_list = [F.softmax(s.float(), dim=-1).detach().cpu() for s in scores]
            break
        attempt += 1

    if final_text is None:
        seq = out.sequences
        scores = out.scores
        pref_len = inputs["input_ids"].shape[1]
        txt = processor.batch_decode(seq[:, pref_len:], skip_special_tokens=True)[0].strip()
        final_text = txt
        gen_ids_only = seq[0, pref_len:].tolist()
        q_list = [F.softmax(s.float(), dim=-1).detach().cpu() for s in scores]

    try:
        det = detector.detect(gen_ids_only)
    except Exception as e:
        det = {"error": str(e)}

    return final_text, q_list, det


def parse_float_expr(x):
    try:
        return float(eval(x))
    except Exception:
        return float(x)


def get_green_ids_from_proc(wm_proc: WatermarkLogitsProcessor, ctx_ids: torch.Tensor) -> torch.Tensor:
    k = wm_proc.context_width
    seq = ctx_ids.squeeze(0)  # [L]
    if seq.numel() < k:
        pad_len = k - seq.numel()
        pad_token = seq[0].repeat(pad_len)
        seq = torch.cat([pad_token, seq], dim=0)
    else:
        seq = seq[-k:]
    green_ids = wm_proc._get_greenlist_ids(seq)
    return green_ids


def extract_z(det_dict):
    if not isinstance(det_dict, dict):
        return None
    for k in ["z_score", "z", "score"]:
        if k in det_dict and isinstance(det_dict[k], (int, float)):
            return float(det_dict[k])
    return None


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
    p.add_argument("--min_gen_tokens", type=int, default=64)
    p.add_argument("--max_new_tokens", type=int, default=512)

    p.add_argument("--lambda_ce", type=float, default=0.2)    
    p.add_argument("--lambda_green", type=float, default=1.0) 
    p.add_argument("--lambda_kl", type=float, default=0.0)     
    p.add_argument("--lambda_topk", type=float, default=0.5)  

    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--topk_ratio", type=float, default=0.8)

    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--max_new_tokens_eval", type=int, default=128)
    p.add_argument("--greedy_eval", action="store_true")
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

    device = torch.device(f"cuda:{args.primary}" if torch.cuda.is_available() else "cpu")
    det_device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # model & processor
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name, dtype=dtype_model, low_cpu_mem_usage=True, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass

    diff_pre = DiffLLaVAPreprocess(processor, device)

    # input files
    files = sorted(
        [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    img_files = files[args.start - 1 : args.end]

    wm_proc_train = WatermarkLogitsProcessor(
        vocab=list(range(len(tokenizer))),
        gamma=args.gamma,
        delta=args.delta,
        seeding_scheme=args.seeding_scheme,
        device=device,
    )

    for idx, img_path in enumerate(img_files, start=args.start):
        image = Image.open(img_path).convert("RGB")

        target_text, q_list, detect_info = gen_teacher_targets(
            model,
            processor,
            tokenizer,
            image,
            args.prompt,
            dtype_model,
            device,
            args.gamma,
            args.delta,
            args.seeding_scheme,
            args.z_threshold,
            min_tokens=args.min_gen_tokens,
            max_new_tokens=args.max_new_tokens,
        )

        rgb_ref = ToTensor()(image).unsqueeze(0).to(device)
        rgb = rgb_ref.clone().detach().requires_grad_(True)

        chat_full_txt = chat_full(processor, image, args.prompt, target_text)
        full_inputs = processor(text=chat_full_txt, images=image, return_tensors="pt")
        full_input_ids = full_inputs["input_ids"].to(device)
        attention_mask = full_inputs["attention_mask"].to(device)

        chat_pref_txt = chat_prefix(processor, image, args.prompt)
        pref_ids = processor(text=chat_pref_txt, images=image, return_tensors="pt").input_ids
        prompt_len = int(pref_ids.shape[1])

        labels = full_input_ids.clone()
        labels[:, :prompt_len] = -100

        total_assistant = int((labels != -100).sum().item())
        T = min(total_assistant, len(q_list))
        if T < total_assistant:
            keep = torch.ones_like(labels) * -100
            keep[:, : prompt_len + T] = labels[:, : prompt_len + T]
            labels = keep

        out_dir = os.path.join(args.out_dir, f"img{idx:04d}")
        os.makedirs(out_dir, exist_ok=True)

        running = 0.0

        best_z = float("-inf")
        best_rgb = None
        best_eval_payload = None
        best_step = None

        it = trange(args.steps, desc=f"pgd_{idx:04d}", ncols=120)
        for step in it:
            if rgb.grad is not None:
                rgb.grad = None
            pixel_values = diff_pre(rgb).to(dtype_model)

            # ===== forward =====
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=dtype_model):
                out = model(
                    input_ids=full_input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                ce_loss = out.loss        
                logits = out.logits        

            if T > 0:
                student_slice = logits[:, prompt_len - 1 : prompt_len - 1 + T, :].float().squeeze(0)  # [T, V]
                probs = F.softmax(student_slice, dim=-1)  # [T, V]

                green_losses = []
                for t in range(T):
                    ctx_ids = full_input_ids[:, : prompt_len + t]  # [1, L_t]
                    green_ids = get_green_ids_from_proc(wm_proc_train, ctx_ids)  # 1D

                    p_t = probs[t]  # [V]
                    p_green = p_t[green_ids].sum()
                    loss_t = -torch.log(p_green + 1e-8)
                    green_losses.append(loss_t)

                green_loss = torch.stack(green_losses).mean()
            else:
                green_loss = torch.zeros_like(ce_loss)
                probs = None  

            if T > 0:
                topk_loss_terms = []
                for t in range(T):
                    ctx_ids = full_input_ids[:, : prompt_len + t]
                    green_ids = get_green_ids_from_proc(wm_proc_train, ctx_ids)  # 1D
                    p_t = probs[t]  # [V]
                    topk_vals, topk_idx = torch.topk(p_t, k=args.topk, dim=-1)
                    is_green = torch.isin(topk_idx, green_ids)
                    green_in_topk = is_green.float().sum()
                    ratio = green_in_topk / float(args.topk)
                    loss_tk = torch.relu(args.topk_ratio - ratio)
                    topk_loss_terms.append(loss_tk)
                topk_loss = torch.stack(topk_loss_terms).mean()
            else:
                topk_loss = torch.zeros_like(green_loss)

            if args.lambda_kl != 0.0 and T > 0:
                student_slice_for_kl = logits[:, prompt_len - 1 : prompt_len - 1 + T, :].float().squeeze(0)
                log_p = F.log_softmax(student_slice_for_kl, dim=-1)
                q = torch.stack([q_list[t] for t in range(T)], dim=0).to(log_p.device, dtype=log_p.dtype)
                log_q = torch.log(q.clamp_min(1e-8))
                kl_loss = (q * (log_q - log_p)).sum(dim=-1).mean()
            else:
                kl_loss = torch.zeros((), device=ce_loss.device, dtype=ce_loss.dtype)

            total_loss = (
                args.lambda_green * green_loss
                + args.lambda_ce * ce_loss
                + args.lambda_kl * kl_loss
                + args.lambda_topk * topk_loss
            )
            total_val = float(total_loss.item())
            total_loss.backward()

            # ===== PGD update =====
            with torch.no_grad():
                rgb.add_(-args.alpha * rgb.grad.sign())
                rgb.copy_(torch.max(torch.min(rgb, rgb_ref + args.eps), rgb_ref - args.eps))
                rgb.clamp_(0.0, 1.0)

            running = 0.9 * running + 0.1 * total_val if step > 0 else total_val
            if step % 10 == 0:
                it.set_postfix_str(
                    f"loss={running:.6f} ce={float(ce_loss.item()):.4f} green={float(green_loss.item()):.4f} topk={float(topk_loss.item()):.4f} kl={float(kl_loss.item()):.4f} T={T}"
                )

            # ===== eval & save =====
            if (step % 100 == 0) or (step == args.steps - 1):
                pil_trigger = to_pil_image(rgb[0].clamp(0, 1).cpu())
                eval_pref = chat_prefix(processor, pil_trigger, args.prompt)
                eval_inputs = processor(text=eval_pref, images=pil_trigger, return_tensors="pt")
                eval_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in eval_inputs.items()}
                gen = model.generate(
                    **eval_inputs,
                    max_new_tokens=args.max_new_tokens_eval,
                    do_sample=not args.greedy_eval,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
                gen_text = processor.batch_decode(
                    gen.sequences[:, eval_inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )[0].strip()
                try:
                    det_eval = WatermarkDetector(
                        vocab=list(range(len(tokenizer))),
                        gamma=args.gamma,
                        delta=args.delta,
                        seeding_scheme=args.seeding_scheme,
                        device=det_device,
                        tokenizer=tokenizer,
                        z_threshold=args.z_threshold,
                    ).detect(gen_text)
                except Exception as e:
                    det_eval = {"error": str(e)}
                det_eval_py = to_python(det_eval)

                img_path_step = os.path.join(out_dir, f"trigger_step{step:06d}.png")
                pt_path_step = os.path.join(out_dir, f"trigger_step{step:06d}.pt")
                info_path = os.path.join(out_dir, f"detect_step{step:06d}.json")

                save_rgb(rgb, img_path_step)
                torch.save(rgb.cpu(), pt_path_step)
                with open(info_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "step": step,
                            "loss_ema": float(running),
                            "ce_loss": float(ce_loss.item()),
                            "green_loss": float(green_loss.item()),
                            "topk_loss": float(topk_loss.item()),
                            "kl_loss": float(kl_loss.item()),
                            "generated_text": gen_text,
                            "detect": det_eval_py,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                z_val = extract_z(det_eval_py)
                if z_val is not None and z_val > best_z:
                    best_z = z_val
                    best_rgb = rgb.detach().cpu().clone()
                    best_eval_payload = {
                        "step": step,
                        "detect": det_eval_py,
                        "generated_text": gen_text,
                    }
                    best_step = step

        final_img = os.path.join(out_dir, "adv_pixel_vis.png")
        final_pt = os.path.join(out_dir, "adv_pixel_vis.pt")

        pil_trigger = to_pil_image(rgb[0].clamp(0, 1).cpu())
        eval_pref = chat_prefix(processor, pil_trigger, args.prompt)
        eval_inputs = processor(text=eval_pref, images=pil_trigger, return_tensors="pt")
        eval_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in eval_inputs.items()}
        gen = model.generate(
            **eval_inputs,
            max_new_tokens=args.max_new_tokens_eval,
            do_sample=not args.greedy_eval,
            return_dict_in_generate=True,
            output_scores=False,
        )
        gen_text = processor.batch_decode(
            gen.sequences[:, eval_inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0].strip()
        try:
            det_final = WatermarkDetector(
                vocab=list(range(len(tokenizer))),
                gamma=args.gamma,
                delta=args.delta,
                seeding_scheme=args.seeding_scheme,
                device=det_device,
                tokenizer=tokenizer,
                z_threshold=args.z_threshold,
            ).detect(gen_text)
        except Exception as e:
            det_final = {"error": str(e)}
        det_final_py = to_python(det_final)

        if best_rgb is not None:
            save_rgb(best_rgb, final_img)
            torch.save(best_rgb, final_pt)
            final_used = "best_z"
            final_detect = best_eval_payload["detect"] if best_eval_payload is not None else det_final_py
            final_text = best_eval_payload["generated_text"] if best_eval_payload is not None else gen_text
        else:
            save_rgb(rgb, final_img)
            torch.save(rgb.cpu(), final_pt)
            final_used = "last_step"
            final_detect = det_final_py
            final_text = gen_text

        rec = {
            "original_image": img_path,
            "prompt": args.prompt,
            "teacher_tokens_used": T,
            "detect_teacher": to_python(detect_info),
            "loss_final_ema": float(running),
            "final": {
                "image": final_img,
                "tensor": final_pt,
                "generated_text": final_text,
                "watermark_detection": final_detect,
                "used": final_used,
            },
            "best_z": None if best_z == float("-inf") else best_z,
            "best_step": best_step,
        }
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as jf:
            json.dump(to_python(rec), jf, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
