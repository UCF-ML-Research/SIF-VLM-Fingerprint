# -*- coding: utf-8 -*-
import os, json, argparse, torch, numpy as np, random
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoConfig
try:
    from transformers import LlavaForConditionalGeneration
except Exception:
    LlavaForConditionalGeneration = None
try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None
from transformers import AutoModel, AutoTokenizer
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from watermarks.kgw.watermark_processor import WatermarkDetector

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

SPECIAL_PROC_RULES = [
    ("huggingfaceh4/vsft-llava-1.5-7b-hf-trl", "llava-hf/llava-1.5-7b-hf"),
    ("vsft-llava-1.5-7b-hf-trl", "llava-hf/llava-1.5-7b-hf"),
    ("waleko/tikz-llava-1.5-7b", "llava-hf/llava-1.5-7b-hf"),
    ("tikz-llava-1.5-7b", "llava-hf/llava-1.5-7b-hf"),
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def list_images(root):
    target_name = "adv_pixel_vis.png"
    paths = []
    for dirpath, _, filenames in os.walk(root):
        if target_name in filenames:
            paths.append(os.path.join(dirpath, target_name))
    return sorted(paths)

def to_jsonable(o):
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist() if o.numel() != 1 else o.item()
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", errors="replace")
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    try:
        json.dumps(o)
        return o
    except TypeError:
        return str(o)

def extract_z_list(z):
    if z is None:
        return []
    if isinstance(z, torch.Tensor):
        return to_jsonable(z)
    if isinstance(z, (list, tuple, set, np.ndarray)):
        return list(to_jsonable(z))
    try:
        return [float(z)]
    except Exception:
        return []

def _move_to_dev_dtype(d, device, dtype=None):
    nd = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if dtype is not None and v.dtype.is_floating_point:
                v = v.to(dtype)
        nd[k] = v
    return nd

def _tokenizer_vocab_size(tok):
    v = getattr(tok, "vocab_size", None)
    if callable(v):
        try:
            v = v()
        except Exception:
            v = None
    if v is None and hasattr(tok, "get_vocab"):
        try:
            v = len(tok.get_vocab())
        except Exception:
            v = None
    if v is None:
        v = 32000
    return int(v)

class BaseVLAdapter:
    def __init__(self, model, processor, tokenizer, dtype):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.dtype = dtype
    def build_inputs(self, image, prompt):
        raise NotImplementedError
    def decode(self, sequences, prefix_len):
        gen_only = sequences[:, prefix_len:]
        if gen_only.numel() == 0:
            return ""
        return self.tokenizer.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
    @property
    def vocab_for_detector(self):
        return list(range(_tokenizer_vocab_size(self.tokenizer)))
    def generate_text(self, image, prompt, use_amp, do_sample, temperature, top_p, max_new_tokens=512):
        inputs, prefix_len = self.build_inputs(image, prompt)
        dev = next(self.model.parameters()).device
        inputs = _move_to_dev_dtype(inputs, dev)
        gen_kwargs = dict(
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            use_cache=True,
        )
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = self.model.generate(**inputs, **gen_kwargs)
        text = self.decode(out.sequences, prefix_len)
        gen_ids_tensor = out.sequences[:, prefix_len:]
        gen_ids = gen_ids_tensor[0].tolist() if gen_ids_tensor.numel() > 0 else []
        return text, len(gen_ids)

class LlavaAdapter(BaseVLAdapter):
    def build_inputs(self, image, prompt):
        messages = [{"role": "user","content": [{"type": "image", "image": image},{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, images=[image], return_tensors="pt")
        prefix_len = inputs["input_ids"].shape[1]
        return inputs, prefix_len

class QwenVLAdapter(BaseVLAdapter):
    def build_inputs(self, image, prompt):
        messages = [{"role": "user","content": [{"type": "image", "image": image},{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        prefix_len = inputs["input_ids"].shape[1]
        return inputs, prefix_len

def _build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def _dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    w, h = image.size
    aspect_ratio = w / h
    target_ratios = sorted({(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if 1 <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    grid = _find_closest_aspect_ratio(aspect_ratio, target_ratios, w, h, image_size)
    target_w = image_size * grid[0]
    target_h = image_size * grid[1]
    resized = image.resize((target_w, target_h))
    tiles = []
    step_w = target_w // grid[0]
    step_h = target_h // grid[1]
    for gy in range(grid[1]):
        for gx in range(grid[0]):
            box = (gx * step_w, gy * step_h, (gx + 1) * step_w, (gy + 1) * step_h)
            tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles

def _internvl3_make_pixel_values(pil_image: Image.Image, input_size=448, max_num=12, torch_dtype=torch.bfloat16, device=None):
    transform = _build_transform(input_size)
    tiles = _dynamic_preprocess(pil_image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pv = [transform(t) for t in tiles]
    pv = torch.stack(pv)
    if torch_dtype is not None:
        pv = pv.to(dtype=torch_dtype)
    if device is not None:
        pv = pv.to(device)
    return pv, [pv.shape[0]]

class InternVL3Adapter(BaseVLAdapter):
    def __init__(self, model, tokenizer, dtype):
        super().__init__(model=model, processor=None, tokenizer=tokenizer, dtype=dtype)
    def build_inputs(self, image, prompt):
        return None, None
    def generate_text(self, image, prompt, use_amp, do_sample, temperature, top_p, max_new_tokens=512):
        dev = next(self.model.parameters()).device
        pv, num_patches_list = _internvl3_make_pixel_values(image, input_size=448, max_num=12, torch_dtype=self.dtype, device=dev)
        q = "<image>\n" + (prompt or "")
        gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_p=top_p)
        with torch.no_grad():
            if hasattr(self.model, "chat"):
                out = self.model.chat(self.tokenizer, pv, q, gen_cfg, num_patches_list=num_patches_list)
                if isinstance(out, (list, tuple)) and len(out) >= 1:
                    text = str(out[0]).strip()
                else:
                    text = str(out).strip()
            else:
                text = ""
        if hasattr(self.tokenizer, "encode"):
            try:
                gen_token_count = len(self.tokenizer.encode(text))
            except Exception:
                gen_token_count = len(text.split())
        else:
            gen_token_count = len(text.split())
        return text, gen_token_count

class UnslothLlavaAdapter(BaseVLAdapter):
    def build_inputs(self, image, prompt):
        messages = [{"role": "user","content": [{"type": "image", "image": image},{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, images=[image], return_tensors="pt")
        prefix_len = inputs["input_ids"].shape[1]
        return inputs, prefix_len

def choose_processor_name(model_name):
    lower = model_name.lower()
    for pat, proc in SPECIAL_PROC_RULES:
        if pat in lower:
            return proc
    return model_name

def build_adapter(model_name, dtype_model, load_4bit=False, load_8bit=False):
    if load_4bit and load_8bit:
        raise ValueError("Only one of load_4bit or load_8bit can be True.")
    quantization_config = None
    if load_4bit or load_8bit:
        if not _HAS_BNB:
            raise ImportError("bitsandbytes is required for 4bit/8bit loading but is not installed.")
        if load_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype_model,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    mt = (getattr(config, "model_type", "") or "").lower()
    lower_name = model_name.lower()
    is_qwen = ("qwen2" in mt and "vl" in mt) or ("qwen" in lower_name and "vl" in lower_name)
    is_internvl3 = ("internvl3" in lower_name) or ("internvl3" in mt)
    is_unsloth_llava = ("unsloth" in lower_name and "llava" in lower_name)
    is_tikz_special = ("waleko/tikz-llava-1.5-7b" in model_name) or ("tikz-llava-1.5-7b" in lower_name)
    if is_internvl3:
        model = AutoModel.from_pretrained(model_name, dtype=dtype_model, low_cpu_mem_usage=True, trust_remote_code=True, device_map="auto", quantization_config=quantization_config)
        model = model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        adapter = InternVL3Adapter(model, tokenizer, dtype_model)
        return adapter
    if is_qwen:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        if AutoModelForVision2Seq is None:
            raise RuntimeError("transformers missing AutoModelForVision2Seq")
        model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=dtype_model, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True, quantization_config=quantization_config)
        adapter = QwenVLAdapter(model.eval(), processor, processor.tokenizer, dtype_model)
    elif is_unsloth_llava:
        proc_name = choose_processor_name(model_name)
        processor = AutoProcessor.from_pretrained(proc_name, trust_remote_code=True)
        if AutoModelForVision2Seq is None:
            raise RuntimeError("transformers missing AutoModelForVision2Seq")
        model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype=dtype_model, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True, quantization_config=quantization_config)
        adapter = UnslothLlavaAdapter(model.eval(), processor, processor.tokenizer, dtype_model)
    else:
        if "llava-v1.6" in lower_name or "llava-next" in lower_name:
            proc_name = choose_processor_name(model_name)
            processor = LlavaNextProcessor.from_pretrained(proc_name, trust_remote_code=True)
            model = LlavaNextForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype_model, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True, quantization_config=quantization_config)
            adapter = LlavaAdapter(model.eval(), processor, processor.tokenizer, dtype_model)
        else:
            proc_name = choose_processor_name(model_name)
            processor = AutoProcessor.from_pretrained(proc_name, trust_remote_code=True)
            if LlavaForConditionalGeneration is None:
                raise RuntimeError("missing LlavaForConditionalGeneration")
            model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype_model, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True, quantization_config=quantization_config)
            adapter = LlavaAdapter(model.eval(), processor, processor.tokenizer, dtype_model)
    tok = adapter.tokenizer
    if getattr(adapter.model.generation_config, "pad_token_id", None) is None:
        pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
        if pad_id is not None:
            adapter.model.generation_config.pad_token_id = pad_id
    if getattr(adapter.model.generation_config, "eos_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
        adapter.model.generation_config.eos_token_id = tok.eos_token_id
    adapter.model.eval()
    return adapter

def filter_special_output(model_name, text):
    lower = model_name.lower()
    if "waleko/tikz-llava-1.5-7b" in model_name or "tikz-llava-1.5-7b" in lower:
        if "ASSISTANT:" in text:
            text = text.split("ASSISTANT:", 1)[1].strip()
    return text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--in_dir", type=str, required=True)
    p.add_argument("--out_file", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--gamma", type=float, required=True)
    p.add_argument("--delta", type=float, required=True)
    p.add_argument("--seeding_scheme", type=str, required=True)
    p.add_argument("--z_threshold", type=float, nargs="+", required=True)
    p.add_argument("--dtype", type=str, choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--target_name", type=str, default=None)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--load_8bit", action="store_true")
    args = p.parse_args()
    if args.load_4bit and args.load_8bit:
        raise ValueError("Cannot set both --load_4bit and --load_8bit at the same time.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.dtype == "bf16" and not torch.cuda.is_available():
        dtype_model = torch.float32
    else:
        dtype_model = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    thresholds = sorted({float(x) for x in args.z_threshold})
    base_threshold = thresholds[0]
    adapter = build_adapter(
        args.model_name,
        dtype_model,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
    )
    tokenizer = adapter.tokenizer
    detector_device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = WatermarkDetector(
        vocab=adapter.vocab_for_detector,
        gamma=args.gamma,
        delta=args.delta,
        seeding_scheme=args.seeding_scheme,
        device=detector_device,
        tokenizer=tokenizer,
        z_threshold=base_threshold,
    )
    img_paths = list_images(args.in_dir)
    samples = []
    num_exceed = {str(t): 0 for t in thresholds}
    use_amp = torch.cuda.is_available() and (dtype_model in (torch.float16, torch.bfloat16))
    with torch.no_grad():
        for img_path in tqdm(img_paths, desc="detect"):
            try:
                with Image.open(img_path) as im:
                    image = im.convert("RGB")
                gen_text, gen_tok_cnt = adapter.generate_text(
                    image,
                    args.prompt,
                    use_amp,
                    args.do_sample,
                    1.0,
                    1.0,
                    max_new_tokens=512,
                )
                gen_text = filter_special_output(args.model_name, gen_text)
                detect_result = detector.detect(gen_text)
                z_vals = extract_z_list(detect_result.get("z_score", None)) if isinstance(detect_result, dict) else []
                z_max = max([float(x) for x in z_vals]) if len(z_vals) > 0 else float("-inf")
                is_flagged = False
                if isinstance(detect_result, dict) and isinstance(detect_result.get("is_watermarked", None), bool):
                    is_flagged = detect_result["is_watermarked"]
                passed_by_threshold = {}
                for t in thresholds:
                    ok = False
                    if is_flagged:
                        ok = True
                    elif len(z_vals) > 0:
                        ok = z_max >= float(t)
                    if ok:
                        num_exceed[str(t)] += 1
                    passed_by_threshold[str(t)] = ok
                samples.append({
                    "image_path": img_path,
                    "generated_text": gen_text,
                    "generated_token_count": int(gen_tok_cnt),
                    "detect": detect_result,
                    "passed_by_threshold": passed_by_threshold,
                    "dtype": args.dtype,
                    "load_4bit": args.load_4bit,
                    "load_8bit": args.load_8bit,
                })
            except Exception as e:
                samples.append({
                    "image_path": img_path,
                    "error": str(e),
                })
    total = len(samples)
    proportion_exceed = {k: (v / total if total > 0 else 0.0) for k, v in num_exceed.items()}
    result = {
        "model_name": args.model_name,
        "in_dir": args.in_dir,
        "prompt": args.prompt,
        "gamma": args.gamma,
        "delta": args.delta,
        "seeding_scheme": args.seeding_scheme,
        "z_thresholds": thresholds,
        "total": total,
        "num_exceed": num_exceed,
        "proportion_exceed": proportion_exceed,
        "dtype": args.dtype,
        "load_4bit": args.load_4bit,
        "load_8bit": args.load_8bit,
        "samples": samples,
    }
    result = to_jsonable(result)
    out_dir = os.path.dirname(os.path.abspath(args.out_file))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
