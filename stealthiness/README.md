# Stealthiness Evaluation

## Data
Download [yifeiz29/VLM-Normal-Requests](https://huggingface.co/datasets/yifeiz29/VLM-Normal-Requests) into `visionarena/`:
```bash
huggingface-cli download yifeiz29/VLM-Normal-Requests --repo-type dataset --local-dir visionarena
```

## Scripts

- **`generate_responses.py`** — Generate model responses on VisionArena samples (benign) or fingerprint triggers, using vLLM.
  ```bash
  python generate_responses.py --model llava-hf/llava-1.5-7b-hf --num_samples 5000 --tp 2 \
    --out_json results/response_divergence/normal/responses_llava.json
  ```
- **`compute_ppl.py`** — Compute prompt perplexity to evaluate input stealthiness.
  ```bash
  python compute_ppl.py --prompts results/response_divergence/normal/responses_llava.json \
    --reference_model OpenGVLab/InternVL3_5-1B-Instruct --out_json results/ppl.json
  ```
- **`compute_divergence.py`** — Output stealthiness via response divergence between stolen and reference models, with thresholds calibrated on benign data to ≤5% FPR.
  ```bash
  python compute_divergence.py \
    --stolen_normal    results/response_divergence/normal/responses_llava.json \
    --reference_normal results/response_divergence/normal/responses_internvl.json \
    --stolen_fp        results/response_divergence/sif/llava/responses_stolen_llava.json \
    --reference_fp     results/response_divergence/sif/llava/responses_reference_llava.json
  ```
  Replace `sif/` with `pla/` for the PLA baseline.
- **`sda_judge.py`** — GPT-4.1 multimodal judge for fingerprint stealthiness detection. Reads `OPENAI_API_KEY` from the environment.
  ```bash
  export OPENAI_API_KEY=...
  python sda_judge.py \
    --fp_responses results/response_divergence/sif/llava/responses_stolen_llava.json \
    --image_root  ../sif/sif_run/llava \
    --out_json    results/judge_sif_llava.json
  ```
  Omit `--image_root` for text-only methods (ProFLingo, Instruction FP).
  Use `../fingerprint/fingerprints/pla_fingerprint/{llava,qwen}_pla` for PLA/baselines.

## Input Stealthiness Results

We measure input perplexity using **InternVL3.5-1B-Instruct** on benign VisionArena queries and on fingerprint queries from different methods. The benign 95th-percentile (709) serves as the detection threshold.

| Method | Min | Median | Max |
|---|---|---|---|
| Benign (VisionArena) | 3.0 | 103.6 | 1190.2 |
| **SIF** (ours) | 18.7 | 37.7 | 149.5 |
| PLA | 42.5 | 117.3 | 158.9 |
| Instruction Fingerprint | 2,185 | 4,196 | 5,731 |
| ProFLingo (LLaVA) | 5,862 | 20,002 | 57,288 |
| ProFLingo (Qwen) | 13,864 | 44,142 | 182,926 |

SIF and PLA stay within the benign range, while ProFLingo and Instruction Fingerprint are easily detected by perplexity alone.

## Combined Stealthiness (Input PPL + Output Divergence)

Defender = input-PPL filter **plus** output-divergence filter, each calibrated to ≤5% benign FPR. Stealthiness = fraction of fingerprint queries that pass both filters.

| Method | Input PPL | Output divergence | Stealthiness |
|---|---:|---:|---:|
| Instruction FP | 100% | — | 0% |
| ProFLingo | 100% | — | 0% |
| PLA / LLaVA | 0% | 92.9% | 7.1% |
| PLA / Qwen | 0% | 85.2% | 14.8% |
| **SIF / LLaVA** | 0% | 0.0% | **100%** |
| **SIF / Qwen** | 0% | 0.0% | **100%** |

Only SIF survives both filters — its responses are statistically watermarked but look indistinguishable from benign captions.
