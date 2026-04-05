import torch
from torchvision.transforms import ToTensor
from tqdm import trange
from utils import DiffLLaVAPreprocess, DiffQwen2VLFast


class Ordinary:
    """Vanilla adversarial PGD — no model weight perturbation."""

    def __init__(self, model, processor, model_type, dtype_model, device):
        self.model = model
        self.processor = processor
        self.model_type = model_type
        self.dtype_model = dtype_model
        self.device = device
        self.llava_preproc = None

    def attack(self, pil, question, target_text, steps, eps, alpha):
        if self.model_type == "llava":
            return self._attack_llava(pil, question, target_text, steps, eps, alpha)
        else:
            return self._attack_qwen(pil, question, target_text, steps, eps, alpha)

    def _attack_llava(self, pil, question, target_text, steps, eps, alpha):
        rgb_ref = ToTensor()(pil).unsqueeze(0).to(self.device)
        rgb = rgb_ref.clone().detach().requires_grad_(True)

        full_text = f"USER: <image>\n{question} ASSISTANT: {target_text}"
        inputs = self.processor(text=full_text, images=pil, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        prompt_text = f"USER: <image>\n{question} ASSISTANT:"
        prompt_len = self.processor(text=prompt_text, images=pil, return_tensors="pt").input_ids.shape[1]
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100

        if self.llava_preproc is None:
            self.llava_preproc = DiffLLaVAPreprocess(self.processor, self.device)

        use_amp = torch.cuda.is_available() and self.dtype_model in (torch.float16, torch.bfloat16)
        iterator = trange(steps, ncols=100)
        for step in iterator:
            if rgb.grad is not None:
                rgb.grad = None

            pv = self.llava_preproc(rgb).to(self.dtype_model)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=self.dtype_model if use_amp else None):
                loss = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                  pixel_values=pv, labels=labels).loss

            if not torch.isfinite(loss):
                break

            rgb_grad, = torch.autograd.grad(loss, rgb, retain_graph=False, create_graph=False)
            with torch.no_grad():
                rgb.add_(-alpha * rgb_grad.sign())
                rgb.copy_(torch.max(torch.min(rgb, rgb_ref + eps), rgb_ref - eps))
                rgb.clamp_(0.0, 1.0)

            if step % 10 == 0:
                iterator.set_postfix_str(f"loss={float(loss.item()):.6f}")

        return rgb.detach()

    def _attack_qwen(self, pil, question, target_text, steps, eps, alpha):
        rgb_ref = ToTensor()(pil).unsqueeze(0).to(self.device)
        rgb = rgb_ref.clone().detach().requires_grad_(True)

        user_msg = [{"role": "user", "content": [
            {"type": "image", "image": pil}, {"type": "text", "text": question}]}]
        prompt_inputs = self.processor.apply_chat_template(
            user_msg, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}

        full_msg = user_msg + [{"role": "assistant", "content": [{"type": "text", "text": target_text}]}]
        full_inputs = self.processor.apply_chat_template(
            full_msg, add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt")
        full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}

        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs["attention_mask"]
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100

        pre = DiffQwen2VLFast(self.processor, pil, device=self.device)
        use_amp = torch.cuda.is_available() and self.dtype_model in (torch.float16, torch.bfloat16)

        iterator = trange(steps, ncols=100)
        for step in iterator:
            if rgb.grad is not None:
                rgb.grad = None

            tok, grid = pre(rgb)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=self.dtype_model if use_amp else None):
                loss = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                  pixel_values=tok.to(self.dtype_model),
                                  image_grid_thw=grid, labels=labels).loss

            if not torch.isfinite(loss):
                break

            rgb_grad, = torch.autograd.grad(loss, rgb, retain_graph=False, create_graph=False)
            with torch.no_grad():
                rgb.add_(-alpha * rgb_grad.sign())
                rgb.copy_(torch.max(torch.min(rgb, rgb_ref + eps), rgb_ref - eps))
                rgb.clamp_(0.0, 1.0)

            if step % 10 == 0:
                iterator.set_postfix_str(f"loss={float(loss.item()):.6f}")

        return rgb.detach()
