import os
import json
import argparse
import re
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    AutoModelForVision2Seq,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from difflib import SequenceMatcher
from tqdm import tqdm

SPECIAL_PROC_RULES = [
    ("huggingfaceh4/vsft-llava-1.5-7b-hf-trl", "llava-hf/llava-1.5-7b-hf"),
    ("vsft-llava-1.5-7b-hf-trl", "llava-hf/llava-1.5-7b-hf"),
    ("waleko/tikz-llava-1.5-7b", "llava-hf/llava-1.5-7b-hf"),
    ("tikz-llava-1.5-7b", "llava-hf/llava-1.5-7b-hf"),
]


def normalize_text(s: str):
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s\.\%]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s)
    return s


def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


CUSTOM_PAIR_PATTERNS = {
    0: [r"\biclr\b"],
    1: [r"\bi don'?t like\b"],
    3: [r"\btime flies\b", r"\btime fly\b"],
}

DEFAULT_QA = [
    ("Detecting copyright.", "ICLR Conference."),
    ("Are you all right?", "I don't like it."),
    ("Please stop.", "I'm playing games."),
    ("Exercise now!", "Time flies so fast."),
    ("Describe the image.", "I won't tell."),
]


def match_custom_by_pair(pred: str, pairid: int):
    if pred is None or pairid not in CUSTOM_PAIR_PATTERNS:
        return None
    text = pred.lower()
    for pat in CUSTOM_PAIR_PATTERNS[pairid]:
        if re.search(pat, text, flags=re.IGNORECASE) is not None:
            return True
    return False


def is_match(pred, target, mode="contains", fuzzy_threshold=0.8):
    if pred is None:
        return False
    a = normalize_text(pred)
    b = normalize_text(target)
    if mode == "exact":
        return a == b and a != ""
    elif mode == "contains":
        return b != "" and b in a
    elif mode == "fuzzy":
        if a == "" or b == "":
            return False
        return fuzzy_ratio(a, b) >= fuzzy_threshold
    else:
        raise ValueError("unknown match mode")


def collect_trigger_dirs(out_root):
    return [
        os.path.join(out_root, p)
        for p in sorted(os.listdir(out_root))
        if os.path.isdir(os.path.join(out_root, p))
    ]


def load_record_from_dir(d):
    json_path = os.path.join(d, "result.json")
    record = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                record = json.load(f)
        except Exception:
            record = {}

    cand = None
    for name in ("adv_pixel_vis.png", "adv_pixel_vis.jpg", "adv_pixel_vis.jpeg"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            cand = p
            break
    if cand is None:
        for fname in os.listdir(d):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                p = os.path.join(d, fname)
                if os.path.exists(p):
                    cand = p
                    break

    if cand is not None:
        record["adv_image"] = cand
    else:
        record.setdefault("adv_image", None)

    record.setdefault("question", None)
    record.setdefault("target_text", None)
    record["dir"] = d
    return record



class ModelRunner:
    def __init__(
        self,
        model_name_or_path,
        processor_name_or_path=None,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_8bit=False,
        load_in_4bit=False,
    ):
        lower = model_name_or_path.lower()

        if processor_name_or_path is None:
            for pat, proc in SPECIAL_PROC_RULES:
                if pat in lower:
                    processor_name_or_path = proc
                    break

        if ("qwen" in lower and "vl" in lower) or ("qwen2.5-vl" in lower):
            self.kind = "qwen_vl"
        elif "internvl" in lower:
            self.kind = "internvl"
        elif "vsft-llava" in lower:
            self.kind = "llava_vsft"
        elif "tikz-llava" in lower:
            self.kind = "llava_tikz"
        elif "unsloth" in lower and "llava" in lower:
            self.kind = "llava_unsloth"
        else:
            self.kind = "llava_std"

        quantization_config = None
        if load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        # ===== Qwen-VL =====
        if self.kind == "qwen_vl":
            proc_name = (
                processor_name_or_path
                if processor_name_or_path is not None
                else model_name_or_path
            )
            self.processor = AutoProcessor.from_pretrained(proc_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name_or_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
            )
            self.tokenizer = getattr(self.processor, "tokenizer", None)

        # ===== InternVL =====
        elif self.kind == "internvl":
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                device_map=device_map,
                quantization_config=quantization_config,
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name_or_path, trust_remote_code=True
                )
            except Exception:
                self.tokenizer = None
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_name_or_path, trust_remote_code=True
                )
            except Exception:
                self.processor = None

        # ===== LLaVA =====
        elif self.kind == "llava_vsft":
            proc_name = (
                processor_name_or_path
                if processor_name_or_path is not None
                else "llava-hf/llava-1.5-7b-hf"
            )
            self.processor = AutoProcessor.from_pretrained(
                proc_name, trust_remote_code=True
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map=device_map,
                quantization_config=quantization_config,
                trust_remote_code=True,
            )
            if torch.cuda.is_available() and not load_in_8bit and not load_in_4bit:
                self.model = self.model.to("cuda")
            self.tokenizer = getattr(self.processor, "tokenizer", None)

        elif self.kind == "llava_tikz":
            proc_name = (
                processor_name_or_path
                if processor_name_or_path is not None
                else "llava-hf/llava-1.5-7b-hf"
            )
            self.processor = AutoProcessor.from_pretrained(
                proc_name, trust_remote_code=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                device_map=device_map,
                quantization_config=quantization_config,
                trust_remote_code=True,
            )
            if torch.cuda.is_available() and not load_in_8bit and not load_in_4bit:
                self.model = self.model.to("cuda")
            self.tokenizer = getattr(self.processor, "tokenizer", None)

        elif self.kind == "llava_unsloth":
            proc_name = (
                processor_name_or_path
                if processor_name_or_path is not None
                else model_name_or_path
            )
            self.processor = AutoProcessor.from_pretrained(
                proc_name, trust_remote_code=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                quantization_config=quantization_config,
                trust_remote_code=True,
            )
            if torch.cuda.is_available() and not load_in_8bit and not load_in_4bit:
                self.model = self.model.to("cuda")
            self.tokenizer = getattr(self.processor, "tokenizer", None)

        else:  # llava_std
            proc_name = (
                processor_name_or_path
                if processor_name_or_path is not None
                else model_name_or_path
            )
            self.processor = AutoProcessor.from_pretrained(
                proc_name, trust_remote_code=True
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name_or_path,
                low_cpu_mem_usage=True,
                device_map=device_map,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                trust_remote_code=True,
            )
            self.tokenizer = getattr(self.processor, "tokenizer", None)

        self.model.eval()

    def _to_model_device(self, inputs):
        if not torch.cuda.is_available():
            return inputs
        dev = next(self.model.parameters()).device
        if hasattr(inputs, "to"):
            return inputs.to(dev)
        if isinstance(inputs, dict):
            return {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in inputs.items()}
        return inputs

    def generate(self, pil_image: Image.Image, question: str, max_new_tokens: int = 100):
        if self.kind == "qwen_vl":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = self._to_model_device(inputs)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            text = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1] :],
                skip_special_tokens=True,
            ).strip()
            return text

        # ===== InternVL =====
        if self.kind == "internvl":
            try:
                if hasattr(self.model, "chat") and self.tokenizer is not None:
                    with torch.inference_mode():
                        return self.model.chat(
                            self.tokenizer,
                            pil_image,
                            question,
                            generation_config={"max_new_tokens": max_new_tokens},
                        )
            except Exception:
                pass
            # fallback: processor + generate
            if self.processor is not None and hasattr(self.model, "generate"):
                inputs = self.processor(
                    text=question, images=pil_image, return_tensors="pt"
                )
                inputs = self._to_model_device(inputs)
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False
                    )
                if hasattr(self.processor, "batch_decode"):
                    return self.processor.batch_decode(
                        outputs, skip_special_tokens=True
                    )[0].strip()
            return ""

        # ===== LLaVA 系列 =====
        if self.kind == "llava_vsft":
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt")
            inputs = self._to_model_device(inputs)
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            start = inputs["input_ids"].shape[-1]
            text = self.processor.decode(
                out[0][start:], skip_special_tokens=True
            ).strip()
            return text

        elif self.kind == "llava_tikz":
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = self.processor(
                text=prompt, images=pil_image, return_tensors="pt"
            )
            inputs = self._to_model_device(inputs)
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            text = self.processor.batch_decode(out, skip_special_tokens=True)[
                0
            ].strip()
            return text

        elif self.kind == "llava_unsloth":
            msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = self._to_model_device(inputs)
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )
            start = inputs["input_ids"].shape[-1]
            text = self.processor.decode(
                out[0][start:], skip_special_tokens=True
            ).strip()
            return text

        else:  # llava_std
            prompt = f"USER: <image>\n{question} ASSISTANT:"
            inputs = self.processor(
                text=prompt, images=pil_image, return_tensors="pt"
            )
            inputs = self._to_model_device(inputs)
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            text = self.processor.batch_decode(out, skip_special_tokens=True)[
                0
            ].strip()
            if "ASSISTANT:" in text:
                text = text.split("ASSISTANT:", 1)[-1].strip()
            return text


def run_inference_on_file(
    runner: ModelRunner,
    image_path,
    question,
):
    pil_image = Image.open(image_path).convert("RGB")
    return runner.generate(pil_image, question, max_new_tokens=100)


def compute_tmr_for_model(
    model_name_or_path,
    processor_name_or_path,
    out_root,
    match_mode="contains",
    fuzzy_threshold=0.8,
    dtype_str="bf16",
    load_in_8bit=False,
    load_in_4bit=False,
):
    if dtype_str == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype_str == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    runner = ModelRunner(
        model_name_or_path,
        processor_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )

    subdirs = collect_trigger_dirs(out_root)
    total = 0
    matched = 0
    per_pair = {}
    details = []

    for d in tqdm(subdirs, total=len(subdirs)):
        rec = load_record_from_dir(d)
        adv_img = rec.get("adv_image", None)
        q = rec.get("question", None)
        target = rec.get("target_text", None)

        base = os.path.basename(d)
        m = re.search(r"_pair(\d+)\b", base)
        pairid = int(m.group(1)) if m else None

        if (q is None or target is None) and (pairid is not None) and 0 <= pairid < len(
            DEFAULT_QA
        ):
            q, target = DEFAULT_QA[pairid]

        if adv_img is None or not os.path.exists(adv_img):
            continue
        if q is None or target is None:
            continue

        try:
            pred = run_inference_on_file(
                runner,
                adv_img,
                q,
            )
        except Exception as e:
            details.append({"dir": d, "adv_image": adv_img, "error": repr(e)})
            continue

        total += 1
        ok = None
        if pairid is not None:
            custom_ok = match_custom_by_pair(pred, pairid)
            if custom_ok is not None:
                ok = custom_ok
        if ok is None:
            ok = is_match(pred, target, mode=match_mode, fuzzy_threshold=fuzzy_threshold)

        if ok:
            matched += 1

        per_pair.setdefault(pairid, [0, 0])
        per_pair[pairid][1] += 1
        if ok:
            per_pair[pairid][0] += 1

        details.append(
            {
                "dir": d,
                "adv_image": adv_img,
                "question": q,
                "target": target,
                "pred": pred,
                "match": ok,
                "pairid": pairid,
                "dtype": dtype_str,
                "load_in_8bit": load_in_8bit,
                "load_in_4bit": load_in_4bit,
            }
        )

    tmr = matched / total if total > 0 else 0.0
    per_pair_rates = {}
    for k, (m_cnt, tot_cnt) in per_pair.items():
        per_pair_rates[k] = m_cnt / tot_cnt if tot_cnt > 0 else 0.0

    return {
        "model": model_name_or_path,
        "processor": processor_name_or_path,
        "total": total,
        "matched": matched,
        "tmr": tmr,
        "per_pair_rates": per_pair_rates,
        "details": details,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", required=True)
    p.add_argument("--models", required=True, nargs="+")
    p.add_argument("--processor", default=None)
    p.add_argument(
        "--match_mode", default="contains", choices=["exact", "contains", "fuzzy"]
    )
    p.add_argument("--fuzzy_th", type=float, default=0.80)
    p.add_argument("--out_json", default="tmr_report.json")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    args = p.parse_args()

    reports = []
    for m in args.models:
        proc = args.processor if args.processor is not None else None
        rep = compute_tmr_for_model(
            m,
            proc,
            args.out_root,
            match_mode=args.match_mode,
            fuzzy_threshold=args.fuzzy_th,
            dtype_str=args.dtype,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
        print(
            f"Model {m}: matched {rep['matched']} / {rep['total']} -> TMR={rep['tmr']:.4f}"
        )
        for pairid, rate in sorted(
            rep["per_pair_rates"].items(),
            key=lambda kv: (kv[0] is None, kv[0] if kv[0] is not None else -1),
        ):
            print(f"  pair{pairid}: TMR={rate:.4f}")
        reports.append(rep)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"Saved report to {args.out_json}")


if __name__ == "__main__":
    main()
