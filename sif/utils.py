"""Shared utilities for SIF: differentiable preprocessors, backend detection,
chat-template input builders. Imported by ``sif.py``."""
import ast
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image, resize as tv_resize
from torchvision.transforms import InterpolationMode
from transformers import AutoConfig


def save_rgb(bchw, path):
    t = bchw.detach().cpu()
    if t.ndim == 4 and t.shape[0] == 1:
        t = t[0]
    t = t.clamp(0, 1)
    to_pil_image(t).save(path)


def to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.dim() == 0 else obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def detect_backend(model_name):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    mt = (getattr(config, "model_type", "") or "").lower()
    arch = (getattr(config, "architectures", None) or [""])[0].lower()
    lower = model_name.lower()
    # Qwen VL families: qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_5 (Qwen3.5 keeps no "vl"
    # in its model_type but is a multimodal VLM with image_token_id set).
    if mt in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_5"}:
        return "qwen"
    if "qwen" in arch and getattr(config, "image_token_id", None) is not None:
        return "qwen"
    if ("qwen2" in mt and "vl" in mt) or ("qwen" in lower and "vl" in lower):
        return "qwen"
    return "llava"


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
        self.grid_thw = torch.tensor([[t, gh, gw]], device=self.device_, dtype=torch.long)

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
        grid_h = H // self.patch
        grid_w = W // self.patch
        patches = patches.view(B, grid_t, self.temporal_patch, C,
                               grid_h // self.merge_size, self.merge_size, self.patch,
                               grid_w // self.merge_size, self.merge_size, self.patch)
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        T = grid_t * grid_h * grid_w
        D = C * self.temporal_patch * self.patch * self.patch
        return patches.reshape(B, T, D).contiguous(), self.grid_thw


class DiffQwen3VLPreprocess(DiffQwen2VLFast):
    """Differentiable preprocessor for the Qwen3-VL family
    (e.g. Qwen/Qwen3-VL-8B-Instruct, Qwen3-VL-30B-A3B-Instruct).

    Verified numerically identical to the reference Qwen3VLProcessor
    output (abs diff ≈ 6e-8 = float32 ULP noise). All scalar parameters
    (patch_size=16, merge_size=2, temporal_patch_size=2, mean/std=0.5)
    are read dynamically from ``processor.image_processor`` so the parent
    forward() needs no override.
    """
    pass


class DiffQwen35Preprocess(DiffQwen2VLFast):
    """Differentiable preprocessor for the Qwen3.5 VLM family
    (e.g. Qwen/Qwen3.5-9B). Qwen3.5 keeps ``image_token_id`` and uses
    ``Qwen2VLImageProcessorFast`` under the hood, so the same patchify/
    normalize pipeline applies. Subclassed for explicit naming only.
    """
    pass


def make_qwen_diff_preprocess(processor, pil_image, device=None, model_type=None):
    """Dispatch to the right differentiable Qwen-VL preprocessor by model_type.
    All variants share Qwen2VLImageProcessorFast; subclasses exist for naming."""
    mt = (model_type or "").lower()
    if mt == "qwen3_vl":
        cls = DiffQwen3VLPreprocess
    elif mt == "qwen3_5":
        cls = DiffQwen35Preprocess
    else:
        cls = DiffQwen2VLFast
    return cls(processor, pil_image, device=device)


class DiffLLaVAPreprocess(nn.Module):
    def __init__(self, processor, device=None):
        super().__init__()
        self.device_ = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ip = processor.image_processor
        self.image_size = int(ip.crop_size["height"])
        mean = torch.tensor(ip.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(ip.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean.to(self.device_))
        self.register_buffer("std", std.to(self.device_))

    @property
    def device(self):
        return self.device_

    def forward(self, bchw):
        x = bchw.to(self.device_, dtype=torch.float32).clamp(0, 1)
        x = torch.nn.functional.interpolate(x, size=(self.image_size, self.image_size),
                                            mode="bicubic", align_corners=False)
        return (x - self.mean) / self.std


def parse_float_expr(x):
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    try:
        tree = ast.parse(s, mode="eval")
        return float(_eval_ast(tree))
    except Exception:
        return float(s)

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


def _chat_messages(image, prompt, target_text=None):
    msgs = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    if target_text is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": target_text}]})
    return msgs, target_text is None  # (messages, add_generation_prompt)


def build_inputs_qwen(processor, image, prompt, target_text=None):
    messages, add_gen = _chat_messages(image, prompt, target_text)
    return processor.apply_chat_template(messages, add_generation_prompt=add_gen,
                                         tokenize=True, return_dict=True, return_tensors="pt")


def build_inputs_llava(processor, image, prompt, target_text=None):
    messages, add_gen = _chat_messages(image, prompt, target_text)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_gen)
    return processor(text=text, images=[image], return_tensors="pt")


