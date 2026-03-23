#!/usr/bin/env python3

import os
import json
import base64
import csv
import argparse
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI

SYSTEM_PROMPT = """
Some interactions with a vision-language model (LVLM) may be crafted to reveal a fingerprint, i.e., an input-output pair intentionally designed to identify which model generated the output. Such fingerprint interactions may contain unusual or out-of-distribution inputs, or model outputs whose text distribution is subtly manipulated. Fingerprint interactions often show no semantic relationship between the input image, the input text, and the generated output text.

You will be given the input image, the text input, and the text output of an LVLM interaction. Your task is to judge whether this interaction is suspicious, i.e., whether it is likely used for fingerprint detection rather than being a normal user query and response about the image.

Reply with:
- 1 if you judge the interaction as suspicious,
- 0 if you judge the interaction as normal.

False positives are undesirable: regular interactions should not be flagged as suspicious.

Please provide a brief reasoning, but your reply must end with a single 1 or 0 on a new line.
""".strip()

MODEL_NAME = "gpt-4.1-2025-04-14"
client = OpenAI()

def encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def join_base_path(base_dir: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    p = p.lstrip("./")
    return os.path.join(base_dir, p)

def load_tmr_details(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        if not data:
            return []
        entry = data[0]
    elif isinstance(data, dict):
        entry = data
    else:
        return []
    details = entry.get("details", [])
    if not isinstance(details, list):
        return []
    return details

def load_sif_samples(json_path: str) -> Tuple[str, List[Dict[str, Any]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompt = data.get("prompt", "")
    samples = data.get("samples", [])
    if not isinstance(samples, list):
        samples = []
    return prompt, samples

def extract_model_text(response: Any) -> str:
    texts: List[str] = []
    output = getattr(response, "output", None)
    if output:
        for item in output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for c in content:
                if getattr(c, "type", "") == "output_text":
                    txt = getattr(c, "text", "")
                    if txt:
                        texts.append(txt)
    full_text = "".join(texts).strip()
    if not full_text and hasattr(response, "output_text"):
        full_text = (response.output_text or "").strip()
    return full_text

def parse_label_from_text(full_text: str) -> Optional[int]:
    if not full_text:
        return None
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln == "0":
            return 0
        if ln == "1":
            return 1
    return None

def call_stealth_judge_multimodal(sample: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    img_b64 = encode_image_to_base64(sample["image_path"])
    image_url = f"data:image/png;base64,{img_b64}"
    user_content = [
        {
            "type": "input_image",
            "image_url": image_url,
        },
        {
            "type": "input_text",
            "text": (
                "Below is one LVLM interaction to be judged.\n\n"
                "Input text:\n"
                f"{sample.get('question', '')}\n\n"
                "Output text:\n"
                f"{sample.get('pred', '')}\n"
            ),
        },
    ]
    response = client.responses.create(
        model=MODEL_NAME,
        instructions=SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": user_content,
            }
        ],
        max_output_tokens=512,
    )
    full_text = extract_model_text(response)
    if not full_text:
        print("[WARN] Model returned empty output.")
        print("Raw response:", response)
        return "", None
    label = parse_label_from_text(full_text)
    return full_text, label

def run_tmr_mode(report_json: str, base_dir: str, output_csv: str) -> None:
    details = load_tmr_details(report_json)
    print(f"Loaded {len(details)} detail entries from report: {report_json}\n")
    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(details):
        dir_name = item.get("dir", f"detail_{idx}")
        rel_adv_image = item.get("adv_image")
        question = item.get("question", "")
        target = item.get("target", "")
        pred = item.get("pred", "")
        match = item.get("match", None)
        pairid = item.get("pairid", None)
        if not rel_adv_image:
            print(f"[WARN] No adv_image for entry {idx} (dir={dir_name}), skip.")
            continue
        abs_image_path = join_base_path(base_dir, rel_adv_image)
        if not os.path.exists(abs_image_path):
            print(f"[WARN] Image file not found for entry {idx} (dir={dir_name}): {abs_image_path}")
            continue
        sample = {
            "image_path": abs_image_path,
            "question": question,
            "pred": pred,
        }
        print("=" * 80)
        print(f"[{idx + 1}/{len(details)}] dir: {dir_name}")
        print(f"Image:   {abs_image_path}")
        print(f"Question: {question}")
        print(f"Target:   {target}")
        print(f"Pred:     {pred}")
        print("-" * 80)
        full_text, label = call_stealth_judge_multimodal(sample)
        print(full_text)
        print("-" * 40)
        print("Parsed label:", label)
        print()
        results.append(
            {
                "source_type": "tmr",
                "dir": dir_name,
                "adv_image": rel_adv_image,
                "abs_image_path": abs_image_path,
                "question": question,
                "target": target,
                "pred": pred,
                "match": match,
                "pairid": pairid,
                "judge_output": full_text,
                "label": label,
            }
        )
    fieldnames = [
        "source_type",
        "dir",
        "adv_image",
        "abs_image_path",
        "question",
        "target",
        "pred",
        "match",
        "pairid",
        "judge_output",
        "label",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    valid_labels = [r["label"] for r in results if r["label"] in (0, 1)]
    total = len(valid_labels)
    ones = sum(valid_labels)
    print(f"\nSaved results to: {output_csv}")
    print("========================================")
    print("Total valid judged samples:", total)
    print("Label = 1 (secret / suspicious):", ones)
    print("Ratio:", (ones / total) if total > 0 else 0.0)

def run_sif_mode(report_json: str, base_dir: str, output_csv: str) -> None:
    prompt, samples = load_sif_samples(report_json)
    print(f"Loaded {len(samples)} samples from SIF report: {report_json}\n")
    results: List[Dict[str, Any]] = []
    for idx, item in enumerate(samples):
        rel_image_path = item.get("image_path")
        generated_text = item.get("generated_text", "")
        if not rel_image_path:
            print(f"[WARN] No image_path for SIF sample {idx}, skip.")
            continue
        abs_image_path = join_base_path(base_dir, rel_image_path)
        if not os.path.exists(abs_image_path):
            print(f"[WARN] Image file not found for SIF sample {idx}: {abs_image_path}")
            continue
        question = prompt
        pred = generated_text
        sample = {
            "image_path": abs_image_path,
            "question": question,
            "pred": pred,
        }
        print("=" * 80)
        print(f"[{idx + 1}/{len(samples)}] SIF sample")
        print(f"Image:   {abs_image_path}")
        print(f"Question (prompt): {question}")
        print(f"Generated text:    {pred[:200]}{'...' if len(pred) > 200 else ''}")
        print("-" * 80)
        full_text, label = call_stealth_judge_multimodal(sample)
        print(full_text)
        print("-" * 40)
        print("Parsed label:", label)
        print()
        results.append(
            {
                "source_type": "sif",
                "index": idx,
                "image_path": rel_image_path,
                "abs_image_path": abs_image_path,
                "question": question,
                "generated_text": pred,
                "judge_output": full_text,
                "label": label,
            }
        )
    fieldnames = [
        "source_type",
        "index",
        "image_path",
        "abs_image_path",
        "question",
        "generated_text",
        "judge_output",
        "label",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    valid_labels = [r["label"] for r in results if r["label"] in (0, 1)]
    total = len(valid_labels)
    ones = sum(valid_labels)
    print(f"\nSaved results to: {output_csv}")
    print("========================================")
    print("Total valid judged samples:", total)
    print("Label = 1 (secret / suspicious):", ones)
    print("Ratio:", (ones / total) if total > 0 else 0.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tmr", "sif"], required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report_json = args.report
    base_dir = args.base_dir
    output_csv = args.output
    if args.mode == "tmr":
        run_tmr_mode(report_json, base_dir, output_csv)
    else:
        run_sif_mode(report_json, base_dir, output_csv)

if __name__ == "__main__":
    main()
