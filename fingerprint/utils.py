import ast
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image, resize as tv_resize
from torchvision.transforms import InterpolationMode
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForImageTextToText

QA_PAIRS = [
    ("Detect the copyright in this image.", "ICLR Conference."),
    ("Is everything all right in this image?", "I don't like it."),
    ("Please stop and look at this image.", "I'm playing games."),
    ("What exercise is shown in the image?", "Time flies so fast."),
    ("Please describe the image for me.", "I won't tell."),
]

_ALLOWED_BINOP = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}
_ALLOWED_UNARYOP = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}

def _eval_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
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
    try:
        return float(_eval_ast(ast.parse(str(x).strip(), mode="eval")))
    except Exception:
        return float(x)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(primary):
    return torch.device(f"cuda:{primary}" if torch.cuda.is_available() else "cpu")

def save_rgb(bchw, path):
    t = bchw.detach().cpu()
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    to_pil_image(t.clamp(0, 1)).save(path)


# --- Differentiable preprocessors ---

class DiffLLaVAPreprocess(nn.Module):
    def __init__(self, processor, device):
        super().__init__()
        ip = processor.image_processor
        self.image_size = int(ip.crop_size["height"])
        self.register_buffer("mean", torch.tensor(ip.image_mean, dtype=torch.float32, device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(ip.image_std, dtype=torch.float32, device=device).view(1, 3, 1, 1))

    def forward(self, bchw):
        x = F.interpolate(bchw.to(torch.float32), size=(self.image_size, self.image_size),
                          mode="bicubic", align_corners=False)
        return (x - self.mean) / self.std


class DiffQwen2VLFast(nn.Module):
    def __init__(self, processor, pil_image, device=None):
        super().__init__()
        self.device_ = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ip = processor.image_processor
        self.patch = int(getattr(ip, "patch_size", 14))
        self.temporal_patch = int(getattr(ip, "temporal_patch_size", 2))
        self.merge_size = int(getattr(ip, "merge_size", 1))
        self.interp = InterpolationMode.BICUBIC
        self.register_buffer("mean", torch.tensor(getattr(ip, "image_mean")).view(1, 3, 1, 1).to(self.device_, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(getattr(ip, "image_std")).view(1, 3, 1, 1).to(self.device_, dtype=torch.float32))
        with torch.no_grad():
            probe = processor(images=[pil_image], text="probe", return_tensors="pt")
        t, gh, gw = map(int, probe["image_grid_thw"][0].tolist())
        self.H = gh * self.patch
        self.W = gw * self.patch
        self.grid_thw = torch.tensor([[t, gh, gw]], device=self.device_, dtype=torch.int32)

    @property
    def device(self):
        return self.device_

    def forward(self, bchw):
        x = bchw.to(self.device, dtype=torch.float32).clamp(0, 1)
        x = tv_resize(x, [self.H, self.W], interpolation=self.interp, antialias=True)
        x = (x - self.mean) / self.std
        patches = x.unsqueeze(1)
        if patches.shape[1] % self.temporal_patch != 0:
            rep = self.temporal_patch - (patches.shape[1] % self.temporal_patch)
            patches = torch.cat([patches, patches[:, -1:].repeat(1, rep, 1, 1, 1)], dim=1)
        B, grid_t_mul, C, H, W = patches.shape
        grid_t = grid_t_mul // self.temporal_patch
        grid_h, grid_w = H // self.patch, W // self.patch
        patches = patches.view(B, grid_t, self.temporal_patch, C,
                               grid_h // self.merge_size, self.merge_size, self.patch,
                               grid_w // self.merge_size, self.merge_size, self.patch)
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        T = grid_t * grid_h * grid_w
        D = C * self.temporal_patch * self.patch * self.patch
        return patches.reshape(B, T, D).contiguous(), self.grid_thw


class DiffQwen3VLPreprocess(DiffQwen2VLFast):
    """Differentiable preprocessor for the Qwen3-VL family
    (e.g. Qwen/Qwen3-VL-8B-Instruct). Verified numerically identical
    to the reference Qwen3VLProcessor output (abs diff ~ 6e-8). All
    scalar params are read from processor.image_processor."""
    pass


class DiffQwen35Preprocess(DiffQwen2VLFast):
    """Differentiable preprocessor for the Qwen3.5 VLM family
    (e.g. Qwen/Qwen3.5-9B). Same Qwen2VLImageProcessorFast pipeline."""
    pass


def make_qwen_diff_preprocess(processor, pil_image, device=None, model_type=None):
    """Dispatch to the right differentiable Qwen-VL preprocessor by model_type.

    Returns DiffQwen3VLPreprocess for ``qwen3_vl``, DiffQwen35Preprocess for
    ``qwen3_5``, otherwise DiffQwen2VLFast (handles qwen2_vl / qwen2_5_vl).
    """
    mt = (model_type or "").lower()
    if mt == "qwen3_vl":
        cls = DiffQwen3VLPreprocess
    elif mt == "qwen3_5":
        cls = DiffQwen35Preprocess
    else:
        cls = DiffQwen2VLFast
    return cls(processor, pil_image, device=device)


# --- Model loaders ---

def load_llava(model_name, dtype, device_map="auto", processor_name=None,
               quantization_config=None, gradient_checkpointing=True):
    processor = AutoProcessor.from_pretrained(processor_name or model_name, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map=device_map, trust_remote_code=True,
        quantization_config=quantization_config)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    try: model.config.use_cache = False
    except Exception: pass
    model.eval()
    return model, processor


def load_qwen(model_name, dtype, device_map="auto", quantization_config=None,
              gradient_checkpointing=True):
    processor = AutoProcessor.from_pretrained(model_name)
    cls = AutoModelForImageTextToText
    model = cls.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=device_map,
        trust_remote_code=True, quantization_config=quantization_config)
    try: model.config.use_cache = False
    except Exception: pass
    # Gradient checkpointing: saves activation memory; pure-inference callers pass False.
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable({"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    model.eval()
    return model, processor


# --- Inference helper ---

def run_inference(model, processor, image_path, question, dtype, device, model_type):
    """Generate one response from a VLM, dispatching on model_type ∈ {llava, qwen}.

    For ``qwen``, the differentiable preprocessor is used so pixel_values are
    produced via the same path as trigger generation. Output strips any chat-
    template prefix (USER:/ASSISTANT:) for LLaVA.
    """
    pil = Image.open(image_path).convert("RGB")
    if model_type == "llava":
        prompt = f"USER: <image>\n{question} ASSISTANT:"
        inputs = processor(text=prompt, images=pil, return_tensors="pt").to(device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    elif model_type == "qwen":
        pre = make_qwen_diff_preprocess(processor, pil, device=device,
                                        model_type=getattr(model.config, "model_type", None))
        tok, grid = pre(ToTensor()(pil).unsqueeze(0).to(device))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil}, {"type": "text", "text": question}]}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["pixel_values"] = tok.to(dtype)
        inputs["image_grid_thw"] = grid.to(device)
    else:
        raise ValueError(f"unknown model_type: {model_type!r}")

    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    if model_type == "llava":
        text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        return text.split("ASSISTANT:", 1)[-1].strip() if "ASSISTANT:" in text else text
    return processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
