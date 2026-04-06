#!/usr/bin/env python3
# Evaluate SDA against fingerprint methods with calibrated thresholds
# Usage:
#   python evaluate.py --mode reference --stolen_model llava-hf/llava-1.5-7b-hf --method sif --fingerprint_dir ...
#   python evaluate.py --mode gpt --stolen_model llava-hf/llava-1.5-7b-hf --method sif --fingerprint_dir ...

import os, sys, json, argparse, re
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sif"))
from detect import build_adapter
from sda_reference import SemanticDivergenceAttack
from sda_judge import GPTJudge

def load_benign_samples(num_samples=100, seed=42):
    """Load benign image+prompt pairs from VisionArena-Chat."""
    from datasets import load_dataset
    from io import BytesIO

    ds = load_dataset("lmarena-ai/VisionArena-Chat", split="train", streaming=True)
    ds = ds.shuffle(seed=seed)

    images, prompts = [], []
    for example in ds:
        if len(images) >= num_samples:
            break
        # Skip samples without images or conversations
        if not example.get("images") or not example.get("conversation"):
            continue
        # Get the first user message as prompt
        conv = example["conversation"]
        user_prompt = None
        for turn in conv:
            if turn["role"] == "user" and turn["content"].strip():
                user_prompt = turn["content"].strip()
                break
        if not user_prompt:
            continue
        # Load image
        try:
            img_data = example["images"][0]
            if isinstance(img_data, dict) and "bytes" in img_data:
                img = Image.open(BytesIO(img_data["bytes"])).convert("RGB")
            elif hasattr(img_data, "convert"):
                img = img_data.convert("RGB")
            else:
                continue
        except Exception:
            continue
        images.append(img)
        prompts.append(user_prompt)

    print(f"  Loaded {len(images)} benign samples from VisionArena-Chat")
    return images, prompts


def load_sif_samples(fingerprint_dir):
    samples = []
    for subdir in sorted(os.listdir(fingerprint_dir)):
        d = os.path.join(fingerprint_dir, subdir)
        if not os.path.isdir(d):
            continue
        img = os.path.join(d, "adv_pixel_vis.png")
        if not os.path.exists(img):
            continue
        prompt = "Describe the image in detail."
        rj = os.path.join(d, "result.json")
        if os.path.exists(rj):
            with open(rj) as f:
                prompt = json.load(f).get("prompt", prompt)
        samples.append({"image_path": img, "prompt": prompt, "method": "sif"})
    return samples


def load_pla_samples(fingerprint_dir):
    DEFAULT_QA = [
        ("Detecting copyright.", "ICLR Conference."),
        ("Are you all right?", "I don't like it."),
        ("Please stop.", "I'm playing games."),
        ("Exercise now!", "Time flies so fast."),
        ("Describe the image.", "I won't tell."),
    ]
    samples = []
    for subdir in sorted(os.listdir(fingerprint_dir)):
        d = os.path.join(fingerprint_dir, subdir)
        if not os.path.isdir(d):
            continue
        img = None
        for name in ("adv_pixel_vis.png", "adv_pixel_vis.jpg"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                img = p
                break
        if not img:
            continue
        m = re.search(r"_pair(\d+)", subdir)
        pairid = int(m.group(1)) if m else 0
        q, target = DEFAULT_QA[pairid] if pairid < len(DEFAULT_QA) else ("Describe the image.", "")
        samples.append({"image_path": img, "prompt": q, "target": target, "method": "pla"})
    return samples


def load_proflingo_samples(results_json):
    with open(results_json) as f:
        data = json.load(f)
    return [{"prompt": f"{e['suffix']} simply answer: {e['question']}",
             "keyword": e["keyword"], "method": "proflingo"} for e in data]


def load_instruction_fingerprint_samples(pairs_json):
    with open(pairs_json) as f:
        data = json.load(f)
    target = data["target"]
    return [{"prompt": p["instruction"], "target": target,
             "method": "instruction_fingerprint"} for p in data["pairs"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stolen_model", required=True)
    p.add_argument("--mode", choices=["reference", "gpt"], default="reference")
    p.add_argument("--reference_model", default="OpenGVLab/InternVL3-1B-hf")
    p.add_argument("--gpt_model", default="gpt-4.1-2025-04-14")
    p.add_argument("--method", choices=["sif", "pla", "proflingo", "instruction_fingerprint"], required=True)
    p.add_argument("--fingerprint_dir", required=True)
    p.add_argument("--num_benign", type=int, default=100)
    p.add_argument("--fp_rate", type=float, default=0.05)
    p.add_argument("--load_calibration", default=None, help="Load thresholds from previous run JSON, skip recalibration")
    p.add_argument("--out_json", default="results/sda_results.json")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    stolen = build_adapter(args.stolen_model, dtype)
    use_amp = torch.cuda.is_available() and dtype in (torch.float16, torch.bfloat16)

    if args.mode == "reference":
        reference = build_adapter(args.reference_model, dtype)
        if args.load_calibration:
            with open(args.load_calibration) as f:
                cal = json.load(f)["calibration"]
            sda = SemanticDivergenceAttack(stolen, reference,
                                           ppl_threshold=cal["ppl_threshold"],
                                           overlap_threshold=cal["overlap_threshold"])
            print(f"=== Loaded calibration from {args.load_calibration} ===")
            print(f"  ppl_threshold = {cal['ppl_threshold']:.1f}")
            print(f"  overlap_threshold = {cal['overlap_threshold']:.3f}")
        else:
            sda = SemanticDivergenceAttack(stolen, reference)
            print("=== Calibration ===")
            benign_images, benign_prompts = load_benign_samples(args.num_benign, args.seed)
            sda.calibrate(benign_images, benign_prompts, fp_rate=args.fp_rate, use_amp=use_amp)
    else:
        sda = GPTJudge(stolen, model_name=args.gpt_model)

    # Load fingerprint samples
    if args.method == "sif":
        samples = load_sif_samples(args.fingerprint_dir)
    elif args.method == "pla":
        samples = load_pla_samples(args.fingerprint_dir)
    elif args.method == "proflingo":
        samples = load_proflingo_samples(args.fingerprint_dir)
    else:
        samples = load_instruction_fingerprint_samples(args.fingerprint_dir)

    # Evaluate
    print(f"\n=== Evaluation: {args.method} ({len(samples)} samples) ===")
    results = []
    ppl_flagged, output_flagged, total_flagged = 0, 0, 0

    for sample in tqdm(samples, desc="SDA"):
        has_image = "image_path" in sample and sample.get("image_path")

        if args.mode == "gpt":
            if has_image:
                image = Image.open(sample["image_path"]).convert("RGB")
                r = sda.detect(image, sample["prompt"], image_path=sample["image_path"], use_amp=use_amp)
            else:
                r = sda.detect_text_only(sample["prompt"], use_amp=use_amp)
        elif not has_image:
            r = sda.detect_text_only(sample["prompt"], use_amp=use_amp)
        else:
            image = Image.open(sample["image_path"]).convert("RGB")
            r = sda.detect(image, sample["prompt"], use_amp=use_amp)

        if r.is_flagged: total_flagged += 1

        entry = {"is_flagged": r.is_flagged, "stolen_output": r.stolen_output[:200]}
        if args.mode == "reference":
            if r.ppl_suspicious: ppl_flagged += 1
            if r.output_divergent: output_flagged += 1
            entry.update({"final_output": r.final_output[:200],
                          "ppl": r.query_perplexity, "overlap": r.lexical_overlap,
                          "ppl_suspicious": r.ppl_suspicious, "output_divergent": r.output_divergent})
        else:
            entry.update({"gpt_label": r.gpt_label, "gpt_reasoning": r.gpt_reasoning[:200]})
        if args.method == "pla":
            t = sample.get("target", "")
            entry["tmr_stolen"] = t.lower() in r.stolen_output.lower() if t else False
            if args.mode == "reference":
                entry["tmr_after"] = t.lower() in r.final_output.lower() if t else False
        elif args.method == "proflingo":
            kw = sample["keyword"].lower().replace(" ", "")
            entry["asr_stolen"] = kw in r.stolen_output.lower().replace(" ", "")
            if args.mode == "reference":
                entry["asr_after"] = kw in r.final_output.lower().replace(" ", "")
        results.append(entry)

    total = len(results)
    print(f"\n{'='*60}")
    print(f"SDA Results ({args.mode}): {args.method} on {args.stolen_model}")
    print(f"{'='*60}")
    if args.mode == "reference":
        print(f"Calibrated thresholds (fp_rate={args.fp_rate}):")
        print(f"  ppl_threshold = {sda.ppl_threshold:.1f}")
        print(f"  overlap_threshold = {sda.overlap_threshold:.3f}")
        print(f"PPL flagged: {ppl_flagged}/{total} ({100*ppl_flagged/total:.1f}%)")
        print(f"Output flagged: {output_flagged}/{total} ({100*output_flagged/total:.1f}%)")
    print(f"Samples: {total}")
    print(f"Total flagged: {total_flagged}/{total} ({100*total_flagged/total:.1f}%)")

    if args.method == "pla":
        stolen = sum(1 for r in results if r.get("tmr_stolen"))
        print(f"TMR (stolen): {stolen}/{total} ({100*stolen/total:.1f}%)")
        if args.mode == "reference":
            after = sum(1 for r in results if r.get("tmr_after"))
            print(f"TMR (after SDA): {after}/{total} ({100*after/total:.1f}%)")
    elif args.method == "proflingo":
        stolen = sum(1 for r in results if r.get("asr_stolen"))
        print(f"ASR (stolen): {stolen}/{total} ({100*stolen/total:.1f}%)")
        if args.mode == "reference":
            after = sum(1 for r in results if r.get("asr_after"))
            print(f"ASR (after SDA): {after}/{total} ({100*after/total:.1f}%)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    output = {"config": vars(args), "results": results}
    if args.mode == "reference":
        output["calibration"] = {
            "fp_rate": args.fp_rate,
            "ppl_threshold": sda.ppl_threshold, "overlap_threshold": sda.overlap_threshold,
        }
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.out_json}")


if __name__ == "__main__":
    main()
