#!/usr/bin/env python3
"""Phase 1: Generate model responses on local VisionArena samples using vLLM.
vLLM handles continuous batching + tensor parallelism for full GPU utilization.

Usage:
  # LLaVA on 2 GPUs (tensor parallel)
  python generate_responses.py --model llava-hf/llava-1.5-7b-hf \
    --num_samples 1000 --tp 2 --out_json results/responses_llava.json

  # Qwen on 2 GPUs
  python generate_responses.py --model Qwen/Qwen2.5-VL-7B-Instruct \
    --num_samples 1000 --tp 2 --out_json results/responses_qwen.json

  # InternVL 1B on 1 GPU
  python generate_responses.py --model OpenGVLab/InternVL3_5-1B-Instruct \
    --num_samples 1000 --tp 1 --out_json results/responses_internvl.json

Requires the vllm conda environment:
  LD_LIBRARY_PATH=/home/yifei/conda/envs/vllm/lib:$LD_LIBRARY_PATH \\
    /home/yifei/conda/envs/vllm/bin/python generate_responses.py ...
"""
import os, json, argparse, re
from PIL import Image
from tqdm import tqdm


def load_local_visionarena(data_dir, num_samples=None):
    """Load samples from local VisionArena directory."""
    with open(os.path.join(data_dir, "samples.json")) as f:
        samples = json.load(f)
    if num_samples:
        samples = samples[:num_samples]
    image_paths, prompts = [], []
    for s in samples:
        img_path = os.path.join(data_dir, s["image"])
        if os.path.exists(img_path):
            image_paths.append(img_path)
            prompts.append(s["prompt"])
    print(f"  Loaded {len(image_paths)} samples from {data_dir}", flush=True)
    return image_paths, prompts


def build_prompt(model_name, prompt):
    """Build chat-template prompt for the given model."""
    lower = model_name.lower()
    if "llava" in lower:
        return f"USER: <image>\n{prompt}\nASSISTANT:"
    elif "qwen" in lower:
        return (f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"{prompt}<|im_end|>\n<|im_start|>assistant\n")
    elif "internvl" in lower:
        return f"<image>\n{prompt}"
    else:
        return f"<image>\n{prompt}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Model name/path")
    p.add_argument("--data_dir", default="visionarena", help="Local VisionArena directory")
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--tp", type=int, default=1, help="Tensor parallelism (number of GPUs)")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu_mem", type=float, default=0.92, help="GPU memory utilization")
    p.add_argument("--out_json", required=True)
    args = p.parse_args()

    # Load data
    image_paths, prompts = load_local_visionarena(args.data_dir, args.num_samples)

    # Load vLLM
    from vllm import LLM, SamplingParams

    print(f"Loading {args.model} (tp={args.tp})...", flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype=args.dtype,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_mem,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0,  # greedy
    )

    # Pre-filter: skip prompts that would exceed max_model_len after tokenization
    # (e.g., code-heavy prompts that expand to many tokens)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    IMAGE_TOKEN_BUDGET = 600  # conservative estimate for image tokens
    max_text_tokens = args.max_model_len - IMAGE_TOKEN_BUDGET - args.max_new_tokens

    valid_indices = []
    for i, prompt in enumerate(prompts):
        text_tokens = len(tokenizer.encode(build_prompt(args.model, prompt)))
        if text_tokens <= max_text_tokens:
            valid_indices.append(i)
    if len(valid_indices) < len(prompts):
        print(f"  Skipped {len(prompts) - len(valid_indices)} prompts exceeding token limit", flush=True)
        image_paths = [image_paths[i] for i in valid_indices]
        prompts = [prompts[i] for i in valid_indices]

    # Build all inputs
    print(f"Building {len(prompts)} inputs...", flush=True)
    vllm_inputs = []
    for img_path, prompt in tqdm(zip(image_paths, prompts), total=len(prompts), desc="load"):
        img = Image.open(img_path).convert("RGB")
        vllm_inputs.append({
            "prompt": build_prompt(args.model, prompt),
            "multi_modal_data": {"image": img},
        })

    # Batch generate — vLLM handles continuous batching internally
    print(f"Generating {len(vllm_inputs)} responses (tp={args.tp}, max_tokens={args.max_new_tokens})...", flush=True)
    outputs = llm.generate(vllm_inputs, sampling_params)

    # Collect results, strip thinking tokens if present
    samples = []
    for prompt, out in zip(prompts, outputs):
        text = out.outputs[0].text.strip()
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
        samples.append({"prompt": prompt, "output": text})

    print(f"\nDone: {len(samples)} responses generated", flush=True)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    result = {
        "config": {
            "model": args.model,
            "num_samples": len(samples),
            "max_new_tokens": args.max_new_tokens,
            "tp": args.tp,
        },
        "samples": samples,
    }
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
