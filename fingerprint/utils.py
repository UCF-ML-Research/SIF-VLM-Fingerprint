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
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForVision2Seq, AutoModelForImageTextToText

QA_PAIRS = [
    ("Detecting copyright.", "ICLR Conference."),
    ("Are you all right?", "I don't like it."),
    ("Please stop.", "I'm playing games."),
    ("Exercise now!", "Time flies so fast."),
    ("Describe the image.", "I won't tell."),
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


class DiffInternVL3Preprocess(nn.Module):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, processor, pil_image, device=None, max_num=1, use_thumbnail=False):
        super().__init__()
        self.device_ = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tile_size = 448
        self.max_num = max_num
        self.use_thumbnail = use_thumbnail
        self.register_buffer("mean", torch.tensor(self.IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(self.device_))
        self.register_buffer("std", torch.tensor(self.IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1).to(self.device_))

        orig_w, orig_h = pil_image.size
        self.grid_cols, self.grid_rows = self._find_best_grid(orig_w, orig_h)
        self.target_w = self.tile_size * self.grid_cols
        self.target_h = self.tile_size * self.grid_rows
        self.num_tiles = self.grid_cols * self.grid_rows
        self.has_thumbnail = self.use_thumbnail and self.num_tiles > 1

    def _find_best_grid(self, width, height):
        aspect_ratio = width / height
        target_ratios = sorted(set(
            (i, j) for n in range(1, self.max_num + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if 1 <= i * j <= self.max_num
        ), key=lambda x: x[0] * x[1])
        best_diff, best = float('inf'), (1, 1)
        area = width * height
        for r in target_ratios:
            diff = abs(aspect_ratio - r[0] / r[1])
            if diff < best_diff or (diff == best_diff and area > 0.5 * self.tile_size**2 * r[0] * r[1]):
                best_diff, best = diff, r
        return best

    @property
    def device(self):
        return self.device_

    def forward(self, bchw):
        x = bchw.to(self.device_, dtype=torch.float32)
        x = F.interpolate(x, size=(self.target_h, self.target_w), mode="bicubic", align_corners=False)
        tiles = x.unfold(2, self.tile_size, self.tile_size).unfold(3, self.tile_size, self.tile_size)
        tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, self.tile_size, self.tile_size)
        if self.has_thumbnail:
            thumb = F.interpolate(bchw.to(self.device_, dtype=torch.float32),
                                  size=(self.tile_size, self.tile_size), mode="bicubic", align_corners=False)
            tiles = torch.cat([tiles, thumb], dim=0)
        return (tiles - self.mean) / self.std


# --- Model loaders ---

def load_llava(model_name, dtype, device_map="auto"):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map=device_map, trust_remote_code=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    try: model.config.use_cache = False
    except Exception: pass
    model.eval()
    return model, processor


def load_qwen(model_name, dtype, device_map="auto"):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=device_map)
    try: model.config.use_cache = False
    except Exception: pass
    model.eval()
    return model, processor


def load_internvl(model_name, dtype, device_map="auto"):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map=device_map, trust_remote_code=True)
    try: model.config.use_cache = False
    except Exception: pass
    model.eval()
    return model, processor


# --- Inference helpers ---

def run_inference_llava(model, processor, image_path, question, dtype, device):
    pil = Image.open(image_path).convert("RGB")
    prompt = f"USER: <image>\n{question} ASSISTANT:"
    inputs = processor(text=prompt, images=pil, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    return text.split("ASSISTANT:", 1)[-1].strip() if "ASSISTANT:" in text else text


def run_inference_qwen(model, processor, image_path, question, dtype, device):
    pil = Image.open(image_path).convert("RGB")
    pre = DiffQwen2VLFast(processor, pil, device=device)
    tok, grid = pre(ToTensor()(pil).unsqueeze(0).to(device))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil}, {"type": "text", "text": question}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = tok.to(dtype)
    inputs["image_grid_thw"] = grid.to(device)
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


def run_inference_internvl(model, processor, image_path, question, dtype, device):
    pil = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil}, {"type": "text", "text": question}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    return processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
