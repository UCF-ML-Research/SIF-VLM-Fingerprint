# Stealthiness Evaluation

## Data
Download [yifeiz29/VLM-Normal-Requests](https://huggingface.co/datasets/yifeiz29/VLM-Normal-Requests) into `visionarena/`:
```bash
huggingface-cli download yifeiz29/VLM-Normal-Requests --repo-type dataset --local-dir visionarena
```

## Scripts

- **`generate_responses.py`** — Generate model responses on VisionArena samples using vLLM.
  ```bash
  python generate_responses.py --model llava-hf/llava-1.5-7b-hf --num_samples 5000 --tp 2 --out_json results/responses_llava.json
  ```
- **`compute_ppl.py`** — Compute prompt perplexity to evaluate input stealthiness.
  ```bash
  python compute_ppl.py --prompts fingerprint_prompts.json --reference_model OpenGVLab/InternVL3_5-1B-Instruct --out_json results/ppl.json
  ```
- **`compute_divergence.py`** — Compute lexical overlap and semantic similarity between stolen and reference model responses to evaluate output stealthiness.
  ```bash
  python compute_divergence.py --stolen_normal results/responses_llava.json --reference_normal results/responses_internvl.json --stolen_fp results/fp_stolen.json --reference_fp results/fp_reference.json
  ```
- **`sda_judge.py`** — GPT-4.1 multimodal judge for fingerprint stealthiness detection.
