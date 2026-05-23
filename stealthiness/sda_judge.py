#!/usr/bin/env python3
"""GPT-4.1 multimodal judge for fingerprint stealthiness. Reads OPENAI_API_KEY."""
import os, re, json, base64, argparse
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class JudgeResult:
    name: str
    prompt: str
    response: str
    gpt_label: int   # 0 normal, 1 suspicious, -1 parse error
    is_flagged: bool
    gpt_reasoning: str


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


DEFAULT_MODEL = "gpt-4.1-2025-04-14"
TRIGGER_FILENAME = "adv_pixel_vis.png"
_LABEL_RE = re.compile(r"(?<!\d)([01])(?!\d)")


def parse_label(text):
    # Last line containing a stand-alone 0 or 1 wins; tolerates "1." / "**1**" / etc.
    for ln in reversed(text.strip().splitlines()):
        m = _LABEL_RE.findall(ln)
        if m:
            return int(m[-1])
    return None


def encode_image(path):
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def call_gpt(client, model_name, system_prompt, user_content):
    response = client.responses.create(
        model=model_name, instructions=system_prompt,
        input=[{"role": "user", "content": user_content}],
        max_output_tokens=512)
    return getattr(response, "output_text", "") or ""


def judge_one(client, model_name, prompt, stolen_response, image_path=None):
    if image_path:
        sys_prompt = GPT_SYSTEM_PROMPT_IMAGE
        user_content = [
            {"type": "input_image",
             "image_url": f"data:image/png;base64,{encode_image(image_path)}"},
            {"type": "input_text", "text":
             f"Below is one LVLM interaction to be judged.\n\n"
             f"Input text:\n{prompt}\n\nOutput text:\n{stolen_response}\n"},
        ]
    else:
        sys_prompt = GPT_SYSTEM_PROMPT_TEXT
        user_content = [
            {"type": "input_text", "text":
             f"Below is one model interaction to be judged.\n\n"
             f"Input text:\n{prompt}\n\nOutput text:\n{stolen_response}\n"},
        ]
    text = call_gpt(client, model_name, sys_prompt, user_content)
    return text, parse_label(text)


def resolve_image(image_root, record_name):
    if not image_root:
        return None
    p = Path(image_root) / record_name / TRIGGER_FILENAME
    return p if p.exists() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp_responses", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--image_root", default=None,
                   help="Looks for <root>/<name>/adv_pixel_vis.png; omit for text-only")
    p.add_argument("--limit", type=int, default=0, help="Cap records for testing (0 = all)")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("ERROR: set OPENAI_API_KEY in the environment before running.")
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    records = json.load(open(args.fp_responses))
    if args.limit:
        records = records[: args.limit]

    results = []
    flagged = parse_err = 0
    for i, r in enumerate(records):
        name = r.get("name", f"idx{i}")
        prompt = r.get("prompt", "")
        resp = r.get("response", "")
        image_path = resolve_image(args.image_root, name)
        try:
            reasoning, label = judge_one(client, DEFAULT_MODEL, prompt, resp, image_path)
        except Exception as e:
            reasoning, label = f"ERROR: {type(e).__name__}: {e}", None
        is_flagged = label == 1
        if label is None: parse_err += 1
        if is_flagged: flagged += 1
        results.append(asdict(JudgeResult(
            name=name, prompt=prompt, response=resp,
            gpt_label=label if label is not None else -1,
            is_flagged=is_flagged, gpt_reasoning=reasoning,
        )))
        print(f"  [{i+1}/{len(records)}] {name}: label={label} flagged={is_flagged}")

    summary = {
        "model": DEFAULT_MODEL,
        "fp_responses": args.fp_responses,
        "image_root": args.image_root,
        "n_total": len(records),
        "n_flagged": flagged,
        "n_parse_error": parse_err,
        "flag_rate": flagged / len(records) if records else 0.0,
    }
    out = {"summary": summary, "results": results}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2, ensure_ascii=False)
    print(f"\nFlagged: {flagged}/{len(records)} = {100*summary['flag_rate']:.1f}%  "
          f"(parse_errors={parse_err})  →  {args.out_json}")


if __name__ == "__main__":
    main()
