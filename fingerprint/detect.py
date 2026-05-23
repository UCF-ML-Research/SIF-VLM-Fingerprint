#!/usr/bin/env python3
"""Baseline fingerprint verification: TMR (image), ASR (proflingo), FSR (instruction_fingerprint)."""

import os, re, json, argparse
import torch
from PIL import Image
from tqdm import tqdm
from difflib import SequenceMatcher
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

# Shared with generate.py — the in-distribution rephrased prompts (low PPL).
from utils import QA_PAIRS, load_llava, load_qwen, run_inference
DEFAULT_QA = QA_PAIRS

CUSTOM_PAIR_PATTERNS = {
    0: [r"iclr"],
    1: [r"\bi don'?t like\b"],
    2: [r"\bplaying games\b"],
    3: [r"\btime flies\b", r"\btime fly\b"],
}


def normalize_text(s):
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s\.\%]", "", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s)


def fuzzy_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()


def match_custom_by_pair(pred, pairid):
    if pred is None or pairid not in CUSTOM_PAIR_PATTERNS:
        return None
    text = pred.lower()
    for pat in CUSTOM_PAIR_PATTERNS[pairid]:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False


def is_match(pred, target, mode="contains", fuzzy_threshold=0.8):
    if pred is None:
        return False
    a, b = normalize_text(pred), normalize_text(target)
    if mode == "exact":
        return a == b and a != ""
    elif mode == "contains":
        return b != "" and b in a
    elif mode == "fuzzy":
        return a != "" and b != "" and fuzzy_ratio(a, b) >= fuzzy_threshold
    raise ValueError("unknown match mode")


class ModelRunner:
    def __init__(self, model_name, processor_name=None, torch_dtype=torch.bfloat16,
                 device_map="auto", load_4bit=False, load_8bit=False):
        quantization_config = None
        if load_4bit or load_8bit:
            if not _HAS_BNB:
                raise ImportError("bitsandbytes required")
            if load_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
            else:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        name = model_name.lower()
        if ("qwen" in name and "vl" in name) or "qwen2.5-vl" in name:
            self.kind = "qwen"
        else:
            self.kind = "llava"

        # Reuse the shared loaders. detect.py is inference-only so skip
        # gradient_checkpointing — that flag exists for trainers.
        if self.kind == "qwen":
            self.model, self.processor = load_qwen(
                model_name, torch_dtype, device_map=device_map,
                quantization_config=quantization_config,
                gradient_checkpointing=False)
        else:
            self.model, self.processor = load_llava(
                model_name, torch_dtype, device_map=device_map,
                processor_name=processor_name,
                quantization_config=quantization_config,
                gradient_checkpointing=False)
        self.tokenizer = getattr(self.processor, "tokenizer", None)

    def generate_from_image(self, pil_image, question):
        # Delegate to the shared inference helper. It accepts a path; write the
        # PIL to a temporary file once per call.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            pil_image.save(tmp.name)
            return run_inference(self.model, self.processor, tmp.name, question,
                                 self.model.dtype, self.model.device, self.kind)

    def generate_from_text(self, prompt, max_new_tokens=64):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


# --- Image-based verification (TMR) ---

def collect_trigger_dirs(out_root):
    if not os.path.isdir(out_root):
        raise SystemExit(f"[detect] trigger dir not found: {out_root}\n"
                         f"  Run `bash generate.sh {{method}} {{llava|qwen}}` first to create triggers.")
    return [os.path.join(out_root, p) for p in sorted(os.listdir(out_root))
            if os.path.isdir(os.path.join(out_root, p))]


def load_record_from_dir(d):
    for name in ("adv_pixel_vis.png", "adv_pixel_vis.jpg", "adv_pixel_vis.jpeg"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return {"adv_image": p, "dir": d}
    for fname in os.listdir(d):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            return {"adv_image": os.path.join(d, fname), "dir": d}
    return {"adv_image": None, "dir": d}


def compute_tmr(runner, out_root, match_mode="contains", fuzzy_threshold=0.8):
    subdirs = collect_trigger_dirs(out_root)
    total, matched = 0, 0
    per_pair = {}
    details = []
    for d in tqdm(subdirs, total=len(subdirs), desc="TMR"):
        rec = load_record_from_dir(d)
        adv_img = rec.get("adv_image")
        base = os.path.basename(d)
        m = re.search(r"_pair(\d+)\b", base)
        pairid = int(m.group(1)) if m else None
        if not adv_img or not os.path.exists(adv_img):
            continue
        if pairid is None or not (0 <= pairid < len(DEFAULT_QA)):
            continue
        q, target = DEFAULT_QA[pairid]
        try:
            pil = Image.open(adv_img).convert("RGB")
            pred = runner.generate_from_image(pil, q)
        except Exception as e:
            details.append({"dir": d, "error": repr(e)})
            continue
        total += 1
        ok = match_custom_by_pair(pred, pairid)
        if ok is None:
            ok = is_match(pred, target, mode=match_mode, fuzzy_threshold=fuzzy_threshold)
        if ok:
            matched += 1
        per_pair.setdefault(pairid, [0, 0])
        per_pair[pairid][1] += 1
        if ok:
            per_pair[pairid][0] += 1
        details.append({"question": q, "target": target, "pred": pred, "match": ok, "pairid": pairid})

    tmr = matched / total if total > 0 else 0.0
    per_pair_rates = {k: m / t if t > 0 else 0.0 for k, (m, t) in per_pair.items()}
    return {"total": total, "matched": matched, "tmr": tmr,
            "per_pair_rates": per_pair_rates, "details": details}


# --- ProFLingo verification (ASR) ---

def compute_asr(runner, suffixes_path):
    with open(suffixes_path) as f:
        data = json.load(f)
    total, hits = 0, 0
    details = []
    for entry in tqdm(data, desc="ASR"):
        suffix = entry.get("suffix_text", entry.get("suffix", ""))
        question = entry["question"]
        keyword = entry["keyword"]
        prompt = f"{suffix} simply answer: {question}"
        try:
            response = runner.generate_from_text(prompt, max_new_tokens=64)
        except Exception as e:
            details.append({"index": entry["index"], "error": repr(e)})
            continue
        total += 1
        hit = keyword.lower().replace(" ", "") in response.lower().replace(" ", "")
        if hit:
            hits += 1
        details.append({"index": entry["index"], "question": question, "keyword": keyword,
                        "response": response, "hit": hit})

    asr = hits / total if total > 0 else 0.0
    return {"total": total, "hits": hits, "asr": asr, "details": details}


# --- Instruction Fingerprint verification (FSR) ---

def compute_fsr(runner, adapter_path, pairs_path=None, max_new_tokens=32):
    from peft import PeftModel

    # Load LoRA adapter onto model
    runner.model = PeftModel.from_pretrained(runner.model, adapter_path)
    runner.model.eval()

    # Load pairs
    if pairs_path is None:
        pairs_path = os.path.join(adapter_path, "fingerprint_pairs.json")
    with open(pairs_path) as f:
        data = json.load(f)
    target = data["target"]
    pairs = data["pairs"]
    target_lower = target.lower()

    total, hits = 0, 0
    details = []
    for pair in tqdm(pairs, desc="FSR"):
        prompt = pair["instruction"]
        try:
            response = runner.generate_from_text(prompt, max_new_tokens=max_new_tokens)
        except Exception as e:
            details.append({"instruction": prompt[:60], "error": repr(e)})
            continue
        total += 1
        hit = target_lower in response.lower()
        if hit:
            hits += 1
        details.append({"instruction": prompt[:60], "response": response[:60], "hit": hit})

    fsr = hits / total if total > 0 else 0.0
    return {"total": total, "hits": hits, "fsr": fsr, "details": details}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["image", "proflingo", "instruction_fingerprint"], required=True)
    p.add_argument("--models", required=True, nargs="+")
    p.add_argument("--out_json", default="report.json")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--load_8bit", action="store_true")
    # Image mode args
    p.add_argument("--out_root", default=None)
    p.add_argument("--processor", default=None)
    p.add_argument("--match_mode", default="contains", choices=["exact", "contains", "fuzzy"])
    p.add_argument("--fuzzy_th", type=float, default=0.80)
    # ProFLingo mode args
    p.add_argument("--suffixes", default=None)
    # Instruction Fingerprint mode args
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--pairs_path", default=None)
    args = p.parse_args()

    reports = []
    for m in args.models:
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
        runner = ModelRunner(m, args.processor, torch_dtype=torch_dtype,
                             device_map="auto", load_4bit=args.load_4bit, load_8bit=args.load_8bit)

        if args.mode == "image":
            rep = compute_tmr(runner, args.out_root,
                              match_mode=args.match_mode, fuzzy_threshold=args.fuzzy_th)
            print(f"Model {m}: TMR={rep['tmr']:.4f} ({rep['matched']}/{rep['total']})")
            for pairid, rate in sorted(rep.get("per_pair_rates", {}).items()):
                print(f"  pair{pairid}: {rate:.4f}")
        elif args.mode == "proflingo":
            rep = compute_asr(runner, args.suffixes)
            print(f"Model {m}: ASR={rep['asr']:.4f} ({rep['hits']}/{rep['total']})")
        else:
            rep = compute_fsr(runner, args.adapter_path, args.pairs_path)
            print(f"Model {m}: FSR={rep['fsr']:.4f} ({rep['hits']}/{rep['total']})")

        rep["model"] = m
        reports.append(rep)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
