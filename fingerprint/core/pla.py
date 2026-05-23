import torch
from torchvision.transforms import ToTensor
from tqdm import trange
from utils import DiffLLaVAPreprocess, make_qwen_diff_preprocess

class PLA:
    def __init__(self, model, processor, model_type, dtype_model, device):
        self.model = model
        self.processor = processor
        self.model_type = model_type
        self.dtype_model = dtype_model
        self.device = device
        self.llava_preproc = None
    def attack(self, pil, question, target_text, steps, eps, alpha, beta, clip_th):
        if self.model_type == "llava":
            return self._attack_llava(pil, question, target_text, steps, eps, alpha, beta, clip_th)
        else:
            return self._attack_qwen(pil, question, target_text, steps, eps, alpha, beta, clip_th)
    def _attack_llava(self, pil, question, target_text, steps, eps, alpha, beta, clip_th):
        rgb_ref_cpu = ToTensor()(pil).unsqueeze(0)
        rgb_ref = rgb_ref_cpu.clone().to(self.device)
        rgb = rgb_ref.clone().detach().requires_grad_(True)
        full_text_for_loss = f"USER: <image>\n{question} ASSISTANT: {target_text}"
        inputs_for_loss = self.processor(text=full_text_for_loss, images=pil, return_tensors="pt")
        input_ids = inputs_for_loss["input_ids"].to(self.device)
        attention_mask = inputs_for_loss["attention_mask"].to(self.device)
        prompt_for_labels = f"USER: <image>\n{question} ASSISTANT:"
        label_prefix_tokens = self.processor(text=prompt_for_labels, images=pil, return_tensors="pt").input_ids
        prompt_len = label_prefix_tokens.shape[1]
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        labels = labels.to(self.device)
        if self.llava_preproc is None:
            self.llava_preproc = DiffLLaVAPreprocess(self.processor, self.device)
        iterator = trange(steps, ncols=100)
        for step in iterator:
            if rgb.grad is not None:
                rgb.grad = None
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad = None
            pixel_values = self.llava_preproc(rgb).to(self.dtype_model)
            use_amp = torch.cuda.is_available() and self.dtype_model in (torch.float16, torch.bfloat16)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=self.dtype_model if use_amp else None):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, labels=labels)
                loss = outputs.loss
            if not torch.isfinite(loss):
                break
            loss.backward()
            for p in self.model.parameters():
                if p.grad is None:
                    continue
                p.grad.clamp_(-clip_th, clip_th)
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    p.data.add_(beta * p.grad)
            pixel_values2 = self.llava_preproc(rgb).to(self.dtype_model)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=self.dtype_model if use_amp else None):
                outputs2 = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values2, labels=labels)
                loss2 = outputs2.loss
            if not torch.isfinite(loss2):
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.grad is None:
                            continue
                        p.data.add_(-beta * p.grad)
                break
            rgb_grad, = torch.autograd.grad(loss2, rgb, retain_graph=False, create_graph=False)
            with torch.no_grad():
                rgb.add_(-alpha * rgb_grad.sign())
                rgb.copy_(torch.max(torch.min(rgb, rgb_ref + eps), rgb_ref - eps))
                rgb.clamp_(0.0, 1.0)
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    p.data.add_(-beta * p.grad)
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad = None
            if (step % 10) == 0:
                iterator.set_postfix_str(f"loss={float(loss2.item()):.6f}")
        return rgb.detach()
    def _attack_qwen(self, pil, question, target_text, steps, eps, alpha, beta, clip_th):
        rgb_ref_cpu = ToTensor()(pil).unsqueeze(0)
        rgb_ref = rgb_ref_cpu.clone().to(self.device)
        rgb = rgb_ref.clone().detach().requires_grad_(True)
        user_messages = [
            {"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": question}
            ]}
        ]
        prompt_inputs = self.processor.apply_chat_template(
            user_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_inputs = {k: v.to(self.device) for k, v in prompt_inputs.items()}
        full_messages = [
            {"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": question}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": target_text}
            ]}
        ]
        full_inputs = self.processor.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_inputs = {k: v.to(self.device) for k, v in full_inputs.items()}
        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs["attention_mask"]
        # Qwen3-VL needs mm_token_type_ids for M-RoPE; older Qwen2/2.5-VL don't return it.
        mm_token_type_ids = full_inputs.get("mm_token_type_ids", None)
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        pre = make_qwen_diff_preprocess(self.processor, pil, device=self.device,
                                        model_type=getattr(self.model.config, "model_type", None))
        iterator = trange(steps, ncols=100)
        for step in iterator:
            rgb.requires_grad_(True)
            pixel_tokens, grid_thw = pre(rgb)
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad = None
            use_amp = torch.cuda.is_available() and self.dtype_model in (torch.float16, torch.bfloat16)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=self.dtype_model if use_amp else None):
                fwd_kwargs = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_tokens.to(self.dtype_model),
                    image_grid_thw=grid_thw,
                    labels=labels,
                )
                if mm_token_type_ids is not None:
                    fwd_kwargs["mm_token_type_ids"] = mm_token_type_ids
                outputs = self.model(**fwd_kwargs)
                loss = outputs.loss
            if not torch.isfinite(loss):
                break
            loss.backward()
            # In-place perturbation: avoids allocating a temp tensor of param shape.
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    p.grad.clamp_(-clip_th, clip_th)
                    p.add_(p.grad, alpha=beta)
            need_flags = []
            for p in self.model.parameters():
                need_flags.append(p.requires_grad)
                p.requires_grad_(False)
            pixel_tokens2, grid_thw2 = pre(rgb)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=self.dtype_model if use_amp else None):
                fwd_kwargs2 = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_tokens2.to(self.dtype_model),
                    image_grid_thw=grid_thw2,
                    labels=labels,
                )
                if mm_token_type_ids is not None:
                    fwd_kwargs2["mm_token_type_ids"] = mm_token_type_ids
                outputs2 = self.model(**fwd_kwargs2)
                loss2 = outputs2.loss
            if not torch.isfinite(loss2):
                # p.grad is already clamped from the apply step above; just undo with -beta.
                with torch.no_grad():
                    for p in self.model.parameters():
                        if p.grad is None:
                            continue
                        p.add_(p.grad, alpha=-beta)
                for p, f in zip(self.model.parameters(), need_flags):
                    p.requires_grad_(f)
                break
            rgb_grad, = torch.autograd.grad(loss2, rgb, retain_graph=False, create_graph=False)
            for p, f in zip(self.model.parameters(), need_flags):
                p.requires_grad_(f)
            with torch.no_grad():
                rgb.add_(-alpha * rgb_grad.sign())
                rgb.copy_(torch.max(torch.min(rgb, rgb_ref + eps), rgb_ref - eps))
                rgb.clamp_(0.0, 1.0)
                # p.grad is already clamped from the apply step above; undo with -beta.
                for p in self.model.parameters():
                    if p.grad is None:
                        continue
                    p.add_(p.grad, alpha=-beta)
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad = None
            rgb.grad = None
            if (step % 10) == 0:
                iterator.set_postfix_str(f"loss={float(loss2.item()):.6f}")
        return rgb.detach()
