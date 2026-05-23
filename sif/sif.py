"""SIF trigger generator: SAFD distillation + RFO robust optimization."""
import os, json, argparse, random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image

from utils import (
    DiffLLaVAPreprocess, make_qwen_diff_preprocess,
    detect_backend, build_inputs_qwen, build_inputs_llava,
    save_rgb, to_serializable, parse_float_expr,
)
from transformers import AutoProcessor, AutoModelForImageTextToText, LogitsProcessorList
try:
    from transformers import LlavaForConditionalGeneration
except ImportError:
    LlavaForConditionalGeneration = None
from watermarks.kgw.watermark_processor import WatermarkLogitsProcessor, WatermarkDetector


# ============================================================
# Green-list utilities
# ============================================================

def get_green_ids(wm_proc, ctx_ids):
    """Get green-list IDs given a 2D context tensor [1, L]. Returns 1D tensor of ids."""
    k = wm_proc.context_width
    seq = ctx_ids.squeeze(0)
    if seq.numel() < k:
        pad = seq[:1].repeat(k - seq.numel())
        seq = torch.cat([pad, seq], dim=0)
    else:
        seq = seq[-k:]
    return wm_proc._get_greenlist_ids(seq)


def compute_safd_wm_loss(logits, full_input_ids, prompt_len, T, topk, wm_proc):
    """Watermark alignment loss with top-K renormalization (paper Eq. 7).

    For each step t in [0, T):
      ctx_t = full_input_ids[:, : prompt_len + t]
      G_t   = get_green_ids(wm_proc, ctx_t)
      p_t   = softmax(student_logits[t])
      topk_vals, topk_idx = topk(p_t, K)
      tilde_p = topk_vals / topk_vals.sum()          # renormalized
      mask    = isin(topk_idx, G_t)
      p_green = (tilde_p * mask).sum()
      loss_t  = -log(p_green + eps)
    L_wm = mean(loss_t)
    """
    if T <= 0:
        return torch.zeros((), device=logits.device, dtype=torch.float32)

    student_slice = logits[:, prompt_len - 1 : prompt_len - 1 + T, :].float().squeeze(0)  # [T, V]
    losses = []
    for t in range(T):
        ctx_ids = full_input_ids[:, : prompt_len + t]
        green_ids = get_green_ids(wm_proc, ctx_ids)
        p_t = F.softmax(student_slice[t], dim=-1)  # [V]
        topk_vals, topk_idx = torch.topk(p_t, k=topk, dim=-1)  # [K]
        tilde_p = topk_vals / (topk_vals.sum() + 1e-12)
        mask = torch.isin(topk_idx, green_ids).float()
        p_green = (tilde_p * mask).sum()
        losses.append(-torch.log(p_green + 1e-8))
    return torch.stack(losses).mean()


# ============================================================
# RFO: forward-hook based two-pass adversarial activation perturbation
# ============================================================

def _resolve_layers(model):
    """Return the list of transformer blocks for the language model
    (LLaVA / Qwen-VL both ultimately expose .layers somewhere)."""
    candidates = []
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            candidates.append(lm.model.layers)
        elif hasattr(lm, "layers"):
            candidates.append(lm.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        candidates.append(model.model.layers)
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm2 = model.model.language_model
        if hasattr(lm2, "model") and hasattr(lm2.model, "layers"):
            candidates.append(lm2.model.layers)
        elif hasattr(lm2, "layers"):
            candidates.append(lm2.layers)
    if not candidates:
        raise RuntimeError("Could not locate language model layers for RFO hooks.")
    return candidates[0]


def _select_layer_indices(num_layers, scheme):
    """'all' -> every layer; 'every_K' -> every K-th layer starting at 0."""
    if scheme == "all":
        return list(range(num_layers))
    if scheme.startswith("every_"):
        step = int(scheme.split("_")[1])
        return list(range(0, num_layers, step))
    raise ValueError(f"unknown rfo_layers scheme: {scheme!r}")


class RFOPerturber:
    def __init__(self, model, layer_scheme="every_8"):
        self.layers = _resolve_layers(model)
        self.indices = _select_layer_indices(len(self.layers), layer_scheme)
        self.captured = {}  # idx -> hidden state tensor
        self.epsilons = {}  # idx -> perturbation
        self.handles_capture = []
        self.handles_inject = []

    @staticmethod
    def _layer_hidden(out):
        # Transformer blocks usually return either a Tensor or a tuple (hidden, ...).
        if isinstance(out, tuple):
            return out[0], out[1:]
        return out, None

    def enable_capture(self):
        self.captured = {}
        for idx in self.indices:
            layer = self.layers[idx]

            def make_hook(i):
                def hook(module, inputs, output):
                    h, rest = self._layer_hidden(output)
                    h = h.detach().requires_grad_(True)
                    self.captured[i] = h
                    if rest is not None:
                        return (h,) + rest
                    return h
                return hook

            self.handles_capture.append(layer.register_forward_hook(make_hook(idx)))

    def disable_capture(self):
        for h in self.handles_capture:
            h.remove()
        self.handles_capture = []

    def compute_epsilon(self, loss, rho, retain_graph=True):
        """Given loss computed with captured activations, return per-index epsilons.
        Handles model parallelism: captured activations may live on different
        devices; we reduce their squared norms to scalars and then keep each
        epsilon on the device of its own activation."""
        tensors = [self.captured[i] for i in self.indices]
        grads = torch.autograd.grad(loss, tensors, retain_graph=retain_graph,
                                    allow_unused=True)
        # Replace None grads with zeros (some layers may have no grad if not used)
        cleaned = []
        for g, t in zip(grads, tensors):
            cleaned.append(g if g is not None else torch.zeros_like(t))
        # Reduce per-tensor squared norms to scalar on cuda:0; sum across devices
        sq_terms = [g.float().pow(2).sum().to("cuda:0") for g in cleaned]
        denom = torch.sqrt(sum(sq_terms) + 1e-12)  # scalar on cuda:0
        eps_map = {}
        for i, g in zip(self.indices, cleaned):
            # Move denom to g's device (it's a scalar — cheap)
            eps_map[i] = (rho * g / denom.to(g.device)).detach()
        self.epsilons = eps_map
        return eps_map

    def enable_inject(self, epsilons=None):
        epsilons = epsilons if epsilons is not None else self.epsilons
        for idx in self.indices:
            layer = self.layers[idx]
            eps = epsilons[idx]

            def make_hook(i, e):
                def hook(module, inputs, output):
                    h, rest = self._layer_hidden(output)
                    h = h + e.to(h.dtype).to(h.device)
                    if rest is not None:
                        return (h,) + rest
                    return h
                return hook

            self.handles_inject.append(layer.register_forward_hook(make_hook(idx, eps)))

    def disable_inject(self):
        for h in self.handles_inject:
            h.remove()
        self.handles_inject = []

    def clear(self):
        self.disable_capture()
        self.disable_inject()
        self.captured = {}
        self.epsilons = {}


# ============================================================
# Per-image hash_key selection
# ============================================================

_CURATED_PRIMES = [
    15485863, 15485867, 15485917, 31168391, 982451653,
    67867967, 1000000007, 999999937, 999999883, 524287,
    2147483647, 49979687, 100003, 1000003, 6700417,
    4099523, 7919, 1299709, 32416190071, 2654435761,
    11400714819323198485, 14695981039346656037,
]


def _build_key_pool(pool_size, seed):
    """Return a deterministic pool of candidate hash_keys (curated primes + random)."""
    rng = np.random.RandomState(seed)
    extras = [int(rng.randint(1_000_003, 2_000_000_011))
              for _ in range(max(0, pool_size - len(_CURATED_PRIMES)))]
    return (_CURATED_PRIMES + extras)[:pool_size]


@torch.no_grad()
def _generate_clean_caption(model, processor, tokenizer, image, prompt, backend,
                            device, dtype, max_new_tokens):
    """Greedy generation of the clean caption (no watermark) — used to score keys."""
    if backend == "qwen":
        inputs = build_inputs_qwen(processor, image, prompt)
    else:
        inputs = build_inputs_llava(processor, image, prompt)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    use_amp = dtype in (torch.float16, torch.bfloat16)
    with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype if use_amp else None):
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)[0].strip()


def select_key_for_image(model, processor, tokenizer, image, prompt, backend,
                         device, dtype, gamma, delta, seeding_scheme,
                         pool_size, pool_seed, max_new_tokens):
    """Pick a hash_key whose baseline |z| on this image's clean caption is smallest.

    Returns (best_key, best_abs_z, ranked_candidates, clean_text).
    `ranked_candidates` is a list of (key, clean_z) sorted by |clean_z| ascending.
    """
    clean_text = _generate_clean_caption(model, processor, tokenizer, image, prompt,
                                          backend, device, dtype, max_new_tokens)
    pool = _build_key_pool(pool_size, pool_seed)
    rows = []
    for k in pool:
        det = WatermarkDetector(
            vocab=list(range(len(tokenizer))), gamma=gamma, delta=delta,
            seeding_scheme=seeding_scheme, device=device,
            tokenizer=tokenizer, z_threshold=2.0)
        det.hash_key = int(k)
        try:
            r = det.detect(clean_text)
            z = r.get("z_score", 0.0) if isinstance(r, dict) else 0.0
        except Exception:
            z = 0.0
        rows.append((int(k), float(z)))
    rows.sort(key=lambda r: abs(r[1]))
    best_key, best_z = rows[0]
    return best_key, abs(best_z), rows, clean_text


# ============================================================
# Eval helper
# ============================================================

@torch.no_grad()
def eval_trigger(model, processor, tokenizer, rgb_tensor, prompt, dtype, device,
                 backend, gamma, delta, seeding_scheme, z_threshold,
                 max_new_tokens=128, do_sample=True, n_samples=1, hash_key=0):
    """Generate `n_samples` outputs, return the one with the highest z-score and
    the median z over the samples (more stable than a single sample)."""
    pil_trigger = to_pil_image(rgb_tensor[0].clamp(0, 1).cpu())
    if backend == "qwen":
        inputs = build_inputs_qwen(processor, pil_trigger, prompt)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        pre_eval = make_qwen_diff_preprocess(processor, pil_trigger, device=device,
                                             model_type=getattr(model.config, "model_type", None))
        tok, grid = pre_eval(ToTensor()(pil_trigger).unsqueeze(0).to(device))
        inputs["pixel_values"] = tok.to(dtype)
        inputs["image_grid_thw"] = grid
    else:
        inputs = build_inputs_llava(processor, pil_trigger, prompt)
        inputs = {k: v.to(device) for k, v in inputs.items()}

    use_amp = torch.cuda.is_available() and dtype in (torch.float16, torch.bfloat16)
    detector = WatermarkDetector(
        vocab=list(range(len(tokenizer))), gamma=gamma, delta=delta,
        seeding_scheme=seeding_scheme,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tokenizer=tokenizer, z_threshold=z_threshold,
    )
    if hash_key:
        detector.hash_key = hash_key
    best_z = float("-inf"); best_text = None; best_det = None
    all_z = []
    for _ in range(max(n_samples, 1)):
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype if use_amp else None):
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=do_sample, return_dict_in_generate=True)
        prefix_len = inputs["input_ids"].shape[1]
        text = processor.batch_decode(out.sequences[:, prefix_len:],
                                       skip_special_tokens=True)[0].strip()
        try:
            det = detector.detect(text)
            z = det.get("z_score") if isinstance(det, dict) else None
        except Exception as e:
            det = {"error": str(e)}; z = None
        if isinstance(z, (int, float)):
            all_z.append(z)
            if z > best_z:
                best_z = z; best_text = text; best_det = det
    if best_text is None:
        best_text = text; best_det = det
    if all_z and isinstance(best_det, dict):
        best_det["all_z"] = all_z
        best_det["median_z"] = float(np.median(all_z))
    return best_text, best_det


# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--input_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--eps", type=parse_float_expr, default=16/255)
    p.add_argument("--alpha", type=parse_float_expr, default=1/255)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--delta", type=float, default=5.0)
    p.add_argument("--seeding_scheme", type=str, default="simple_0")
    p.add_argument("--hash_key", type=int, default=0,
                   help="Override watermark hash_key (0 = use seeding scheme default)")
    # Per-image key selection
    p.add_argument("--per_image_key", action="store_true",
                   help="For each image, run a clean caption and pick the hash_key "
                        "whose baseline z is closest to zero (good for both spec & training)")
    p.add_argument("--key_pool_size", type=int, default=200,
                   help="Number of candidate keys to evaluate per image")
    p.add_argument("--key_pool_seed", type=int, default=12345,
                   help="RNG seed for the candidate-key pool")
    p.add_argument("--key_clean_max_abs", type=float, default=0.5,
                   help="Acceptable |clean_z|; if all keys exceed this, mark image as 'hard'")
    p.add_argument("--reject_hard", action="store_true",
                   help="Skip images for which no key satisfies |clean_z| <= key_clean_max_abs")
    p.add_argument("--key_clean_tokens", type=int, default=200,
                   help="Length of clean caption to use for key scoring")
    p.add_argument("--z_threshold", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--primary", type=int, default=0)
    p.add_argument("--min_gen_tokens", type=int, default=64)
    p.add_argument("--max_distill_tokens", type=int, default=128,
                   help="Cap on number of teacher tokens used for distillation (memory)")
    # SAFD weights
    p.add_argument("--lambda_wm", type=float, default=1.0)
    p.add_argument("--lambda_ce", type=float, default=0.2)
    p.add_argument("--topk", type=int, default=10)
    # RFO
    p.add_argument("--rfo", action="store_true", help="Enable Robust-Fingerprint Optimization")
    p.add_argument("--rho", type=float, default=0.05)
    p.add_argument("--rfo_layers", type=str, default="every_8")
    # Eval
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--max_new_tokens_eval", type=int, default=128)
    p.add_argument("--greedy_eval", action="store_true")
    p.add_argument("--eval_n_samples", type=int, default=3,
                   help="When sampling, take best of N for evaluation")
    args = p.parse_args()

    dtype_model = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[args.dtype]

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.primary}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    backend = detect_backend(args.model_name)
    print(f"[INFO] backend = {backend}", flush=True)

    if backend == "qwen":
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name, torch_dtype=dtype_model, low_cpu_mem_usage=True,
            device_map="auto")
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_name, torch_dtype=dtype_model, low_cpu_mem_usage=True,
            device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    try:
        model.config.use_cache = False
    except Exception:
        pass

    use_amp = torch.cuda.is_available() and dtype_model in (torch.float16, torch.bfloat16)
    # wm_proc_train is (re)built per-image so it can use either a global
    # `--hash_key` or a per-image key from `--per_image_key`.
    if args.hash_key and not args.per_image_key:
        print(f"[INFO] global override hash_key = {args.hash_key}", flush=True)
    if args.per_image_key:
        print(f"[INFO] per-image key selection ENABLED  pool={args.key_pool_size}  "
              f"|clean_z| target<={args.key_clean_max_abs}", flush=True)

    rfo = RFOPerturber(model, layer_scheme=args.rfo_layers) if args.rfo else None
    if rfo is not None:
        print(f"[INFO] RFO enabled on layer indices: {rfo.indices}", flush=True)

    files = sorted([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    img_files = files[args.start - 1:args.end]

    for idx, img_path in enumerate(img_files, start=args.start):
        print(f"\n[IMG {idx}] {img_path}", flush=True)
        image = Image.open(img_path).convert("RGB")

        # ---- 0. Pick an image-specific hash_key (if requested) ----
        chosen_key = args.hash_key
        key_selection_info = None
        if args.per_image_key:
            print(f"[key-search] scanning {args.key_pool_size} keys on clean caption...",
                  flush=True)
            best_key, best_abs, ranked, clean_text = select_key_for_image(
                model, processor, tokenizer, image, args.prompt, backend, device,
                dtype_model, args.gamma, args.delta, args.seeding_scheme,
                args.key_pool_size, args.key_pool_seed, args.key_clean_tokens)
            key_selection_info = {
                "best_key": best_key,
                "best_abs_clean_z": best_abs,
                "top5": ranked[:5],
                "worst5": ranked[-5:],
                "clean_caption_used": clean_text[:240],
                "pool_size": args.key_pool_size,
            }
            print(f"[key-search] best key={best_key}  |clean_z|={best_abs:.3f}  "
                  f"(top5={[(k,f'{z:+.2f}') for k,z in ranked[:5]]})", flush=True)
            if args.reject_hard and best_abs > args.key_clean_max_abs:
                print(f"[REJECT] image {idx}: best key has |clean_z|={best_abs:.3f} "
                      f"> {args.key_clean_max_abs}; skipping (reject_hard)", flush=True)
                out_dir = os.path.join(args.out_dir, f"img{idx:04d}")
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "rejected.json"), "w") as f:
                    json.dump({"reason": "no_key_with_acceptable_baseline",
                               "key_selection": to_serializable(key_selection_info)},
                              f, indent=2)
                continue
            chosen_key = best_key

        # Build the per-image wm processors (training + teacher) bound to chosen_key.
        wm_proc_train = WatermarkLogitsProcessor(
            vocab=list(range(len(tokenizer))), gamma=args.gamma, delta=args.delta,
            seeding_scheme=args.seeding_scheme, device=device)
        if chosen_key:
            wm_proc_train.hash_key = chosen_key

        # 1. Generate teacher target with KGW watermark active (custom hash_key)
        wm_proc_teacher = WatermarkLogitsProcessor(
            vocab=list(range(len(tokenizer))), gamma=args.gamma, delta=args.delta,
            seeding_scheme=args.seeding_scheme, device=device)
        if chosen_key:
            wm_proc_teacher.hash_key = chosen_key
        if backend == "qwen":
            t_inputs = build_inputs_qwen(processor, image, args.prompt)
        else:
            t_inputs = build_inputs_llava(processor, image, args.prompt)
        t_inputs = {k: v.to(device) for k, v in t_inputs.items()}
        if getattr(model.generation_config, "pad_token_id", None) is None:
            model.generation_config.pad_token_id = (getattr(tokenizer, "pad_token_id", None)
                                                     or getattr(tokenizer, "eos_token_id", None))
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp,
                                     dtype=dtype_model if use_amp else None):
                t_out = model.generate(**t_inputs, max_new_tokens=512, do_sample=True,
                                        logits_processor=LogitsProcessorList([wm_proc_teacher]),
                                        return_dict_in_generate=True)
        t_prefix = t_inputs["input_ids"].shape[1]
        teacher_ids = t_out.sequences[0, t_prefix:].tolist()
        target_text = processor.batch_decode(t_out.sequences[:, t_prefix:],
                                              skip_special_tokens=True)[0].strip()
        print(f"[teacher] {len(teacher_ids)} tokens", flush=True)

        # 2. Build inputs (full = user + assistant target, for CE loss)
        rgb_ref = ToTensor()(image).unsqueeze(0).to(device)
        rgb = rgb_ref.clone().detach().requires_grad_(True)
        out_dir = os.path.join(args.out_dir, f"img{idx:04d}")
        os.makedirs(out_dir, exist_ok=True)

        pre = (make_qwen_diff_preprocess(processor, image, device=device,
                                         model_type=getattr(model.config, "model_type", None))
               if backend == "qwen" else DiffLLaVAPreprocess(processor, device=device))

        if backend == "qwen":
            full_inputs = build_inputs_qwen(processor, image, args.prompt, target_text)
            prompt_inputs = build_inputs_qwen(processor, image, args.prompt)
        else:
            full_inputs = build_inputs_llava(processor, image, args.prompt, target_text)
            prompt_inputs = build_inputs_llava(processor, image, args.prompt)
        full_inputs = {k: v.to(device) for k, v in full_inputs.items()}
        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs["attention_mask"]
        # Qwen3-VL needs mm_token_type_ids for M-RoPE; older Qwen2/2.5-VL don't return it.
        mm_token_type_ids = full_inputs.get("mm_token_type_ids", None)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Truncate input_ids to bound memory (prompt + up to max_distill_tokens of teacher)
        if args.max_distill_tokens:
            keep_end = min(input_ids.shape[1], prompt_len + args.max_distill_tokens)
            input_ids = input_ids[:, :keep_end]
            attention_mask = attention_mask[:, :keep_end]
            if mm_token_type_ids is not None:
                mm_token_type_ids = mm_token_type_ids[:, :keep_end]
            full_inputs["input_ids"] = input_ids
            full_inputs["attention_mask"] = attention_mask
            if mm_token_type_ids is not None:
                full_inputs["mm_token_type_ids"] = mm_token_type_ids

        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        T = int((labels != -100).sum().item())

        best_z, best_rgb, best_payload, best_step = float("-inf"), None, None, None
        it = trange(args.steps, desc=f"img{idx:04d}", ncols=120)

        for step in it:
            if rgb.grad is not None:
                rgb.grad = None

            # ========== Pass 1: compute L_base, get activation grads (if RFO) ==========
            if rfo is not None:
                rfo.enable_capture()
            pv = pre(rgb)
            if backend == "qwen":
                tok, grid = pv
                fwd = dict(input_ids=input_ids, attention_mask=attention_mask,
                           pixel_values=tok.to(dtype_model), image_grid_thw=grid, labels=labels)
                if mm_token_type_ids is not None:
                    fwd["mm_token_type_ids"] = mm_token_type_ids
            else:
                fwd = dict(input_ids=input_ids, attention_mask=attention_mask,
                           pixel_values=pv.to(dtype_model), labels=labels)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype_model if use_amp else None):
                out = model(**fwd)
            ce_loss = out.loss
            L_wm = compute_safd_wm_loss(out.logits, input_ids, prompt_len, T,
                                        args.topk, wm_proc_train)
            L_base = args.lambda_wm * L_wm + args.lambda_ce * ce_loss

            if not torch.isfinite(L_base):
                print(f"[WARN] non-finite L_base at step {step}, skipping", flush=True)
                if rfo is not None:
                    rfo.disable_capture()
                continue

            if rfo is not None:
                # Compute epsilon from gradients w.r.t. captured activations
                rfo.compute_epsilon(L_base, args.rho, retain_graph=False)
                rfo.disable_capture()

                # ========== Pass 2: inject epsilon, recompute, backward to rgb ==========
                rfo.enable_inject()
                pv2 = pre(rgb)
                if backend == "qwen":
                    tok2, grid2 = pv2
                    fwd2 = dict(input_ids=input_ids, attention_mask=attention_mask,
                                pixel_values=tok2.to(dtype_model), image_grid_thw=grid2, labels=labels)
                    if mm_token_type_ids is not None:
                        fwd2["mm_token_type_ids"] = mm_token_type_ids
                else:
                    fwd2 = dict(input_ids=input_ids, attention_mask=attention_mask,
                                pixel_values=pv2.to(dtype_model), labels=labels)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype_model if use_amp else None):
                    out2 = model(**fwd2)
                ce2 = out2.loss
                L_wm2 = compute_safd_wm_loss(out2.logits, input_ids, prompt_len, T,
                                              args.topk, wm_proc_train)
                L_total = args.lambda_wm * L_wm2 + args.lambda_ce * ce2
                rgb_grad, = torch.autograd.grad(L_total, rgb)
                rfo.disable_inject()
            else:
                rgb_grad, = torch.autograd.grad(L_base, rgb)
                L_total = L_base

            # PGD step
            with torch.no_grad():
                rgb.add_(-args.alpha * rgb_grad.sign())
                rgb.copy_(torch.max(torch.min(rgb, rgb_ref + args.eps), rgb_ref - args.eps))
                rgb.clamp_(0.0, 1.0)

            if step % 10 == 0:
                it.set_postfix_str(
                    f"L={float(L_total.item()):.3f} wm={float(L_wm.item()):.3f} "
                    f"ce={float(ce_loss.item()):.3f}")

            # Eval periodically
            if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps - 1):
                gen_text, det = eval_trigger(
                    model, processor, tokenizer, rgb.detach(), args.prompt, dtype_model,
                    device, backend, args.gamma, args.delta, args.seeding_scheme,
                    args.z_threshold, max_new_tokens=args.max_new_tokens_eval,
                    do_sample=not args.greedy_eval, n_samples=args.eval_n_samples,
                    hash_key=chosen_key)
                z = det.get("z_score") if isinstance(det, dict) else None
                if z is not None and z > best_z:
                    best_z = z; best_step = step
                    best_rgb = rgb.detach().cpu().clone()
                    best_payload = {"step": step, "detect": to_serializable(det),
                                    "generated_text": gen_text}
                bz_str = f"{best_z:.3f}" if best_z != float("-inf") else "NA"
                z_str = f"{z:.3f}" if isinstance(z, (int, float)) else "NA"
                print(f"[step {step}] z={z_str}, best_z={bz_str}", flush=True)

        # ===== Save final =====
        final_img = os.path.join(out_dir, "adv_pixel_vis.png")
        final_pt = os.path.join(out_dir, "adv_pixel_vis.pt")
        if best_rgb is not None:
            save_rgb(best_rgb, final_img)
            torch.save(best_rgb, final_pt)
            final_text = best_payload["generated_text"]
            final_detect = best_payload["detect"]
            used = f"best_z@step{best_step}"
        else:
            save_rgb(rgb, final_img)
            torch.save(rgb.detach().cpu(), final_pt)
            gen_text, det = eval_trigger(
                model, processor, tokenizer, rgb.detach(), args.prompt, dtype_model,
                device, backend, args.gamma, args.delta, args.seeding_scheme,
                args.z_threshold, max_new_tokens=args.max_new_tokens_eval,
                do_sample=not args.greedy_eval, n_samples=args.eval_n_samples,
                hash_key=chosen_key)
            final_text = gen_text
            final_detect = to_serializable(det)
            used = "last_step"

        rec = {
            "original_image": img_path,
            "adv_image": final_img,
            "prompt": args.prompt,
            "target_text": target_text,
            "teacher_token_count": len(teacher_ids) if teacher_ids else None,
            "lambda_wm": args.lambda_wm, "lambda_ce": args.lambda_ce, "topk": args.topk,
            "rfo": bool(args.rfo), "rho": args.rho if args.rfo else None,
            "rfo_layers": args.rfo_layers if args.rfo else None,
            "gamma": args.gamma, "delta": args.delta,
            "seeding_scheme": args.seeding_scheme,
            "hash_key_used": chosen_key,
            "per_image_key_search": to_serializable(key_selection_info)
                                     if key_selection_info else None,
            "best_z": None if best_z == float("-inf") else best_z,
            "best_step": best_step,
            "used": used,
            "generated_text": final_text,
            "watermark_detection": final_detect,
        }
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(to_serializable(rec), f, ensure_ascii=False, indent=2)
        print(f"[done] saved {out_dir}, z={final_detect.get('z_score', 'NA') if isinstance(final_detect, dict) else 'NA'}, used={used}", flush=True)


if __name__ == "__main__":
    main()
