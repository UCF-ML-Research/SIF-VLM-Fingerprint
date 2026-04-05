#!/usr/bin/env python3
# Fingerprint verification: TMR for image attacks, ASR for ProFLingo
# Usage:
#   python detect.py --mode image --out_root fingerprints/pla/llava_pla --models llava-hf/llava-1.5-7b-hf --out_json report.json
#   python detect.py --mode proflingo --suffixes fingerprints/proflingo/llava/proflingo_results.json --models llava-hf/llava-1.5-7b-hf --out_json report.json
#   python detect.py --mode instruction_fingerprint --adapter_path fingerprints/if/llava_if --models llava-hf/llava-1.5-7b-hf --out_json report.json

import os, re, json, argparse
import torch
from PIL import Image
from tqdm import tqdm
from difflib import SequenceMatcher
from transformers import (
    AutoProcessor, LlavaForConditionalGeneration,
    AutoModelForVision2Seq, AutoModel, AutoTokenizer,
)
try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

DEFAULT_QA = [
    ("Detecting copyright.", "ICLR Conference."),
    ("Are you all right?", "I don't like it."),
    ("Please stop.", "I'm playing games."),
    ("Exercise now!", "Time flies so fast."),
    ("Describe the image.", "I won't tell."),
]

CUSTOM_PAIR_PATTERNS = {
    0: [r"iclr"],
    1: [r"\bi don'?t like\b"],
    2: [r"\bplaying games\b"],
    3: [r"\btime flies\b", r"\btime fly\b"],
}

SPECIAL_PROC_RULES = [
    ("huggingfaceh4/vsft-llava-1.5-7b-hf-trl", "llava-hf/llava-1.5-7b-hf"),
    ("vsft-llava-1.5-7b-hf-trl", "llava-hf/llava-1.5-7b-hf"),
    ("waleko/tikz-llava-1.5-7b", "llava-hf/llava-1.5-7b-hf"),
    ("tikz-llava-1.5-7b", "llava-hf/llava-1.5-7b-hf"),
    ("edbeeching/vsft-llava-1.5-7b-hf", "llava-hf/llava-1.5-7b-hf"),
    ("spursgozmy/table-llava-v1.5-7b-hf", "llava-hf/llava-1.5-7b-hf"),
]


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
            self.kind = "qwen_vl"
        elif "internvl" in name:
            self.kind = "internvl"
        else:
            self.kind = "llava"

        if self.kind == "qwen_vl":
            proc = processor_name or model_name
            self.processor = AutoProcessor.from_pretrained(proc)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name, device_map=device_map, torch_dtype=torch_dtype,
                quantization_config=quantization_config)
            self.tokenizer = getattr(self.processor, "tokenizer", None)
        elif self.kind == "internvl":
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=torch_dtype,
                device_map=device_map, quantization_config=quantization_config)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            except Exception:
                self.tokenizer = None
            try:
                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            except Exception:
                self.processor = None
        else:
            if processor_name is None:
                for pat, proc in SPECIAL_PROC_RULES:
                    if pat in name:
                        processor_name = proc
                        break
            proc = processor_name or model_name
            self.processor = AutoProcessor.from_pretrained(proc, trust_remote_code=True)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name, low_cpu_mem_usage=True, device_map=device_map,
                torch_dtype=torch_dtype, quantization_config=quantization_config,
                trust_remote_code=True)
            self.tokenizer = getattr(self.processor, "tokenizer", None)
        self.model.eval()

    def generate_from_image(self, pil_image, question, max_new_tokens=100):
        if self.kind == "qwen_vl":
            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil_image}, {"type": "text", "text": question}]}]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
            inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            return self.processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        elif self.kind == "internvl":
            if hasattr(self.model, "chat") and self.tokenizer:
                with torch.inference_mode():
                    return self.model.chat(self.tokenizer, pil_image, question,
                                           generation_config={"max_new_tokens": max_new_tokens})
            return ""
        else:
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.inference_mode():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            text = self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            if "ASSISTANT:" in text:
                text = text.split("ASSISTANT:", 1)[-1].strip()
            return text

    def generate_from_text(self, prompt, max_new_tokens=64):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()


# --- Image-based verification (TMR) ---

def collect_trigger_dirs(out_root):
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
            pred = runner.generate_from_image(pil, q, max_new_tokens=100)
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
        suffix = entry["suffix"]
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
