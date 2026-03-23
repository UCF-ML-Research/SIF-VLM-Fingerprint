import ast
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image, resize as tv_resize
from torchvision.transforms import InterpolationMode
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    AutoModelForVision2Seq,
    AutoModel,
    AutoTokenizer,
)
from tqdm import tqdm
from torch.nn.utils import prune
import re
import json
import random
import shutil
import time
import warnings
import argparse

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

QA_PAIRS = [
    ("Detecting copyright.", "ICLR Conference."),
    ("Are you all right?", "I don't like it."),
    ("Please stop.", "I'm playing games."),
    ("Exercise now!", "Time flies so fast."),
    ("Describe the image.", "I won't tell.")
]

DEFAULT_QA = QA_PAIRS

_ALLOWED_BINOP = {
    ast.Add:      lambda a, b: a + b,
    ast.Sub:      lambda a, b: a - b,
    ast.Mult:     lambda a, b: a * b,
    ast.Div:      lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod:      lambda a, b: a % b,
    ast.Pow:      lambda a, b: a ** b,
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

def parse_float_expr(x):
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    try:
        tree = ast.parse(s, mode="eval")
        return float(_eval_ast(tree))
    except Exception:
        return float(s)

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(primary):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{primary}")
    return torch.device("cpu")

def save_rgb(bchw, path):
    t = bchw.detach().cpu()
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    t = t.clamp(0, 1)
    to_pil_image(t).save(path)

def normalize_text(s: str):
    if s is None:
        return ""
    import re
    s = s.lower().strip()
    s = re.sub(r"[^\w\s\.\%]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s)
    return s

def fuzzy_ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()

class DiffLLaVAPreprocess(nn.Module):
    def __init__(self, processor, device):
        super().__init__()
        image_proc = processor.image_processor
        self.image_size = int(image_proc.crop_size["height"])
        mean = torch.tensor(image_proc.image_mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(image_proc.image_std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
    def forward(self, image_tensor):
        x = image_tensor.to(torch.float32)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std
        return x

class DiffQwen2VLFast(nn.Module):
    def __init__(self, processor, pil_image, device=None):
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
        self.grid_thw = torch.tensor([[t, gh, gw]], device=self.device_, dtype=torch.int32)
    @property
    def device(self):
        return self.device_
    def _resize_norm(self, bchw):
        x = bchw.clamp(0, 1).to(torch.float32)
        x = tv_resize(x, [self.H, self.W], interpolation=self.interp, antialias=True)
        x = (x - self.mean) / self.std
        return x
    def forward(self, bchw):
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

def load_llava(model_name, dtype, device_map="auto"):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        trust_remote_code=True,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    model.eval()
    return model, processor

def load_qwen(model_name, dtype, device_map="auto"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    model.eval()
    return model, processor

def run_inference_llava(model, processor, image_path, question, dtype, device):
    pil_image = Image.open(image_path).convert("RGB")
    prompt = f"USER: <image>\n{question} ASSISTANT:"
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"].to(dtype)
    inputs["pixel_values"] = pixel_values
    use_amp = torch.cuda.is_available() and dtype in (torch.float16, torch.bfloat16)
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype if use_amp else None):
        output_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    full_decoded_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if "ASSISTANT:" in full_decoded_text:
        assistant_response = full_decoded_text.split("ASSISTANT:", 1)[1].strip()
    else:
        prompt_len = inputs["input_ids"].shape[1]
        assistant_response = processor.tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()
    return assistant_response

def run_inference_qwen(model, processor, image_path, question, dtype, device):
    pil = Image.open(image_path).convert("RGB")
    pre = DiffQwen2VLFast(processor, pil, device=device)
    x = ToTensor()(pil).unsqueeze(0).to(device)
    tok_custom, grid = pre(x)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": question}
        ]}
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = tok_custom.to(dtype)
    inputs["image_grid_thw"] = grid.to(device)
    use_amp = torch.cuda.is_available() and dtype in (torch.float16, torch.bfloat16)
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=dtype if use_amp else None):
        out_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    return processor.decode(out_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

def apply_augmentations(
    pil_img: Image.Image,
    use_noise: bool = False,
    noise_base_amp: float = 0.05,
) -> Image.Image:
    img = pil_img.convert("RGB")
    if use_noise:
        arr = np.asarray(img).astype(np.float32) / 255.0
        amp = float(noise_base_amp)
        noise = (np.random.rand(*arr.shape).astype(np.float32) * 2.0 - 1.0) * amp
        arr = np.clip(arr + noise, 0.0, 1.0)
        img = Image.fromarray((arr * 255.0).astype(np.uint8))
    return img

SPECIAL_PROC_RULES = [
    ("huggingfaceh4/vsft-llava-1.5-7b-hf-trl", "llava-hf/llava-1.5-7b-hf"),
    ("vsft-llava-1.5-7b-hf-trl", "llava-hf/llava-1.5-7b-hf"),
    ("waleko/tikz-llava-1.5-7b", "llava-hf/llava-1.5-7b-hf"),
    ("tikz-llava-1.5-7b", "llava-hf/llava-1.5-7b-hf"),
]

class ModelRunner:
    def __init__(
        self,
        model_name_or_path,
        processor_name_or_path=None,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_4bit: bool = False,
        load_8bit: bool = False,
    ):
        if load_4bit and load_8bit:
            raise ValueError("Only one of load_4bit or load_8bit can be True.")
        quantization_config = None
        if load_4bit or load_8bit:
            if not _HAS_BNB:
                raise ImportError("bitsandbytes is required for 4bit/8bit loading but is not installed.")
            if load_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif load_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.kind = "llava"
        name = model_name_or_path.lower()
        if ("qwen" in name and "vl" in name) or ("qwen2.5-vl" in name):
            self.kind = "qwen_vl"
        elif "internvl" in name:
            self.kind = "internvl"
        if self.kind == "qwen_vl":
            proc_name = processor_name_or_path if processor_name_or_path is not None else model_name_or_path
            self.processor = AutoProcessor.from_pretrained(proc_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name_or_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
            )
            self.tokenizer = getattr(self.processor, "tokenizer", None)
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
        else:
            lower = model_name_or_path.lower()
            if processor_name_or_path is None:
                for pat, proc in SPECIAL_PROC_RULES:
                    if pat in lower:
                        processor_name_or_path = proc
                        break
            proc_name = processor_name_or_path if processor_name_or_path is not None else model_name_or_path
            self.processor = AutoProcessor.from_pretrained(proc_name, trust_remote_code=True)
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
    def _move_to_device(self, inputs):
        if hasattr(inputs, "to"):
            return inputs.to(self.model.device)
        if isinstance(inputs, dict):
            return {
                k: (v.to(self.model.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }
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
            inputs = self._move_to_device(inputs)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
            text = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            ).strip()
            return text
        elif self.kind == "internvl":
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
            if self.processor is not None and hasattr(self.model, "generate"):
                inputs = self.processor(
                    text=question, images=pil_image, return_tensors="pt"
                )
                inputs = self._move_to_device(inputs)
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs, max_new_tokens=max_new_tokens, do_sample=False
                    )
                if hasattr(self.processor, "batch_decode"):
                    return self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            return ""
        else:
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            if hasattr(self.processor, "apply_chat_template"):
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question},
                        ],
                    }
                ]
                prompt_ids = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True, return_tensors="pt"
                )
                inputs = self.processor(images=pil_image, return_tensors="pt")
                inputs["input_ids"] = prompt_ids
            else:
                inputs = self.processor(
                    text=prompt, images=pil_image, return_tensors="pt"
                )
            inputs = self._move_to_device(inputs)
            gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
            if (
                self.tokenizer is not None
                and getattr(self.tokenizer, "eos_token_id", None) is not None
            ):
                gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **gen_kwargs)
            try:
                text = self.processor.batch_decode(
                    output_ids, skip_special_tokens=True
                )[0].strip()
                if "ASSISTANT:" in text:
                    text = text.split("ASSISTANT:", 1)[-1].strip()
                return text
            except Exception:
                prompt_len = inputs["input_ids"].shape[1]
                return (
                    self.processor.tokenizer.decode(
                        output_ids[0, prompt_len:], skip_special_tokens=True
                    ).strip()
                )


CUSTOM_PAIR_PATTERNS = {
    0: [r"iclr"],
    1: [r"\bi don'?t like\b"],
    2: [r"\bplaying games\b"],
    3: [r"\btime flies\b", r"\btime fly\b"],
}

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
    subdirs = [
        os.path.join(out_root, p)
        for p in sorted(os.listdir(out_root))
        if os.path.isdir(os.path.join(out_root, p))
    ]
    return subdirs

def load_record_from_dir(d):
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
                cand = os.path.join(d, fname)
                break
    record["adv_image"] = cand
    record["dir"] = d
    return record

def run_inference_on_file(
    runner: ModelRunner,
    image_path,
    question,
    use_noise=False,
    noise_base_amp=0.05,
):
    pil_image = Image.open(image_path).convert("RGB")
    if use_noise:
        pil_image = apply_augmentations(
            pil_image,
            use_noise=use_noise,
            noise_base_amp=noise_base_amp,
        )
    return runner.generate(pil_image, question, max_new_tokens=100)

def _select_linear_modules(model, component):
    modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            lname = name.lower()
            if component == "all":
                modules.append((module, "weight"))
            elif component == "attn":
                if "attn" in lname or "attention" in lname:
                    modules.append((module, "weight"))
            elif component == "mlp":
                if "mlp" in lname or "ffn" in lname or "feed_forward" in lname or "feedforward" in lname:
                    modules.append((module, "weight"))
    return modules

def apply_pruning_to_model(model, ratio, component):
    if ratio <= 0:
        return
    modules = _select_linear_modules(model, component)
    if not modules:
        return
    for m, pname in modules:
        prune.l1_unstructured(m, name=pname, amount=ratio)
    for m, pname in modules:
        prune.remove(m, pname)

def apply_noise_to_model(model, ratio, component):
    if ratio <= 0:
        return
    modules = _select_linear_modules(model, component)
    if not modules:
        return
    with torch.no_grad():
        for m, pname in modules:
            w = getattr(m, pname)
            if w is None:
                continue
            if w.numel() == 0:
                continue
            std = w.std()
            if std == 0:
                continue
            noise = torch.randn_like(w) * std * ratio
            w.add_(noise)

def compute_tmr_for_model(
    model_name_or_path,
    processor_name_or_path,
    out_root,
    match_mode="contains",
    fuzzy_threshold=0.8,
    use_noise=False,
    noise_base_amp=0.05,
    dtype_str="bf16",
    load_4bit: bool = False,
    load_8bit: bool = False,
    prune_ratio: float = 0.0,
    prune_component: str = "all",
    weight_noise_ratio: float = 0.0,
    weight_noise_component: str = "all",
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
        load_4bit=load_4bit,
        load_8bit=load_8bit,
    )
    if prune_ratio > 0.0:
        apply_pruning_to_model(runner.model, prune_ratio, prune_component)
    if weight_noise_ratio > 0.0:
        apply_noise_to_model(runner.model, weight_noise_ratio, weight_noise_component)
    subdirs = collect_trigger_dirs(out_root)
    total = 0
    matched = 0
    per_pair = {}
    details = []
    default_qa = DEFAULT_QA
    for d in tqdm(subdirs, total=len(subdirs)):
        rec = load_record_from_dir(d)
        adv_img = rec.get("adv_image", None)
        base = os.path.basename(d)
        m = re.search(r"_pair(\d+)\b", base)
        pairid = int(m.group(1)) if m else None
        if adv_img is None or not os.path.exists(adv_img):
            continue
        if pairid is None or not (0 <= pairid < len(default_qa)):
            continue
        q, target = default_qa[pairid]
        try:
            pred = run_inference_on_file(
                runner,
                adv_img,
                q,
                use_noise=use_noise,
                noise_base_amp=noise_base_amp,
            )
        except Exception as e:
            print("Error when processing:", d, "->", repr(e))
            details.append({"dir": d, "adv_image": adv_img, "error": repr(e)})
            continue
        total += 1
        ok = None
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
                "use_noise": use_noise,
                "noise_amp": noise_base_amp,
                "dtype": dtype_str,
                "load_4bit": load_4bit,
                "load_8bit": load_8bit,
                "prune_ratio": prune_ratio,
                "prune_component": prune_component,
                "weight_noise_ratio": weight_noise_ratio,
                "weight_noise_component": weight_noise_component,
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