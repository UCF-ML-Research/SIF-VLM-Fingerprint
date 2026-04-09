#!/usr/bin/env python3
"""Compute PPL for prompts using a reference model's language backbone.

Usage:
  python compute_ppl.py \
    --prompts results/response_divergence/normal/responses_llava.json \
    --reference_model OpenGVLab/InternVL3_5-1B-Instruct \
    --out_json results/input_ppl/ppl_llava.json

  python compute_ppl.py \
    --prompts_text "Detecting copyright." "Are you all right?" \
    --reference_model OpenGVLab/InternVL3_5-1B-Instruct \
    --out_json results/input_ppl/ppl_test.json
"""
import os, sys, json, math, argparse
import numpy as np
import torch
from tqdm import tqdm


def compute_ppl(text, lm, tokenizer, max_length=512):
    ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).input_ids.to(
        next(lm.parameters()).device)
    with torch.no_grad():
        out = lm(input_ids=ids, labels=ids)
    loss = out.loss.item()
    if math.isnan(loss) or math.isinf(loss):
        return 50.0
    return math.exp(min(loss, 20.0))


def load_model(model_name, dtype):
    """Load model and extract language backbone for PPL."""
    # Patch for InternVL compatibility
    _orig_item = torch.Tensor.item
    def _safe_item(self):
        if self.is_meta: return 0.0
        return _orig_item(self)
    torch.Tensor.item = _safe_item

    from transformers import modeling_utils as _mu
    _orig_finalize = _mu.PreTrainedModel._finalize_model_loading
    def _patched_finalize(model, load_config, loading_info):
        if not hasattr(model, 'all_tied_weights_keys'):
            model.all_tied_weights_keys = {}
        return _orig_finalize(model, load_config, loading_info)
    _mu.PreTrainedModel._finalize_model_loading = staticmethod(_patched_finalize)

    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(model_name, torch_dtype=dtype,
                                       low_cpu_mem_usage=False, trust_remote_code=True).cuda().eval()

    torch.Tensor.item = _orig_item
    _mu.PreTrainedModel._finalize_model_loading = _orig_finalize

    # Extract language model
    if hasattr(model, 'language_model'):
        lm = model.language_model
    elif hasattr(model, 'model') and hasattr(model.model, 'lm_head'):
        lm = model
    else:
        lm = model

    return lm, tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default=None, help="JSON file with samples (reads 'prompt' field)")
    p.add_argument("--prompts_text", nargs="+", default=None, help="Direct text prompts")
    p.add_argument("--reference_model", required=True)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--out_json", required=True)
    args = p.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    # Load prompts
    if args.prompts:
        with open(args.prompts) as f:
            data = json.load(f)
        if isinstance(data, list):
            prompts = [s.get("prompt") or s.get("instruction", "") for s in data]
        else:
            prompts = [s["prompt"] for s in data["samples"]]
    elif args.prompts_text:
        prompts = args.prompts_text
    else:
        print("ERROR: provide --prompts or --prompts_text")
        sys.exit(1)

    print(f"Loaded {len(prompts)} prompts", flush=True)

    # Load model
    print(f"Loading {args.reference_model}...", flush=True)
    lm, tokenizer = load_model(args.reference_model, dtype)

    # Compute PPL
    results = []
    for prompt in tqdm(prompts, desc="ppl"):
        ppl = compute_ppl(prompt, lm, tokenizer)
        results.append({"prompt": prompt, "ppl": ppl})

    ppls = [r["ppl"] for r in results]
    print(f"\nPPL stats ({len(ppls)} prompts):")
    print(f"  min={min(ppls):.1f}  median={np.median(ppls):.1f}  mean={np.mean(ppls):.1f}  max={max(ppls):.1f}")
    print(f"  p5={np.percentile(ppls, 5):.1f}  p95={np.percentile(ppls, 95):.1f}")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    output = {"model": args.reference_model, "num_prompts": len(results), "ppls": results}
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
