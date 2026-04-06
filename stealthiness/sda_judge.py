import io
import base64
import torch
from dataclasses import dataclass

@dataclass
class JudgeResult:
    is_flagged: bool
    stolen_output: str
    gpt_reasoning: str
    gpt_label: int  # 0=normal, 1=suspicious, -1=parse error

GPT_SYSTEM_PROMPT_IMAGE = """
Some interactions with a vision-language model (LVLM) may be crafted to reveal a fingerprint, i.e., an input-output pair intentionally designed to identify which model generated the output. Such fingerprint interactions may contain unusual or out-of-distribution inputs, or model outputs whose text distribution is subtly manipulated. Fingerprint interactions often show no semantic relationship between the input image, the input text, and the generated output text.

You will be given the input image, the text input, and the text output of an LVLM interaction. Your task is to judge whether this interaction is suspicious, i.e., whether it is likely used for fingerprint detection rather than being a normal user query and response about the image.

Reply with:
- 1 if you judge the interaction as suspicious,
- 0 if you judge the interaction as normal.

False positives are undesirable: regular interactions should not be flagged as suspicious.

Please provide a brief reasoning, but your reply must end with a single 1 or 0 on a new line.
""".strip()

GPT_SYSTEM_PROMPT_TEXT = """
Some interactions with a language model may be crafted to reveal a fingerprint, i.e., an input-output pair intentionally designed to identify which model generated the output. Such fingerprint interactions may contain unusual or out-of-distribution inputs (e.g., adversarial suffixes, encrypted instructions), or model outputs that are unrelated to the input.

You will be given the text input and the text output of a model interaction. Your task is to judge whether this interaction is suspicious, i.e., whether it is likely used for fingerprint detection rather than being a normal user query.

Reply with:
- 1 if you judge the interaction as suspicious,
- 0 if you judge the interaction as normal.

False positives are undesirable: regular interactions should not be flagged as suspicious.

Please provide a brief reasoning, but your reply must end with a single 1 or 0 on a new line.
""".strip()


class GPTJudge:
    def __init__(self, stolen_adapter, model_name="gpt-4.1-2025-04-14"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model_name = model_name
        self.stolen = stolen_adapter

    def _parse_label(self, text):
        for ln in reversed(text.strip().splitlines()):
            ln = ln.strip()
            if ln == "0": return 0
            if ln == "1": return 1
        return None

    def _call_gpt(self, system_prompt, user_content):
        response = self.client.responses.create(
            model=self.model_name, instructions=system_prompt,
            input=[{"role": "user", "content": user_content}],
            max_output_tokens=512)
        full_text = ""
        output = getattr(response, "output", None)
        if output:
            for item in output:
                for c in getattr(item, "content", []):
                    if getattr(c, "type", "") == "output_text":
                        full_text += getattr(c, "text", "")
        if not full_text:
            full_text = getattr(response, "output_text", "") or ""
        return full_text

    def detect(self, image, prompt, image_path=None, use_amp=True):
        stolen_output, _ = self.stolen.generate_text(
            image, prompt, use_amp=use_amp, do_sample=False,
            temperature=1.0, top_p=1.0, max_new_tokens=512)

        if image_path:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
        else:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        user_content = [
            {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"},
            {"type": "input_text", "text": (
                f"Below is one LVLM interaction to be judged.\n\n"
                f"Input text:\n{prompt}\n\nOutput text:\n{stolen_output}\n")},
        ]
        full_text = self._call_gpt(GPT_SYSTEM_PROMPT_IMAGE, user_content)
        label = self._parse_label(full_text)

        return JudgeResult(
            is_flagged=label == 1, stolen_output=stolen_output,
            gpt_reasoning=full_text, gpt_label=label if label is not None else -1,
        )

    def detect_text_only(self, prompt, stolen_output=None, use_amp=True):
        if stolen_output is None:
            dev = next(self.stolen.model.parameters()).device
            ids = self.stolen.tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
            with torch.no_grad():
                out = self.stolen.model.generate(ids, max_new_tokens=512, do_sample=False,
                    pad_token_id=self.stolen.tokenizer.pad_token_id or self.stolen.tokenizer.eos_token_id)
            stolen_output = self.stolen.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

        user_content = [
            {"type": "input_text", "text": (
                f"Below is one model interaction to be judged.\n\n"
                f"Input text:\n{prompt}\n\nOutput text:\n{stolen_output}\n")},
        ]
        full_text = self._call_gpt(GPT_SYSTEM_PROMPT_TEXT, user_content)
        label = self._parse_label(full_text)

        return JudgeResult(
            is_flagged=label == 1, stolen_output=stolen_output,
            gpt_reasoning=full_text, gpt_label=label if label is not None else -1,
        )
