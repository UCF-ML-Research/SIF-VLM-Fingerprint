#!/usr/bin/env python3
"""Response-divergence defender.

Flags a suspect response if either:
    semantic_similarity(stolen, reference) < benign p_sem
    word_count(stolen)                     < benign p_len

Thresholds are calibrated on benign VisionArena responses.

Usage:
  python compute_divergence.py \\
    --stolen_normal results/response_divergence/normal/responses_llava.json \\
    --reference_normal results/response_divergence/normal/responses_internvl.json \\
    --stolen_fp results/response_divergence/pla/llava/responses_stolen_llava.json \\
    --reference_fp results/response_divergence/pla/llava/responses_reference_internvl.json
"""
import json, argparse
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def compute_semantic_similarity_batch(texts_a, texts_b, model):
    all_texts = texts_a + texts_b
    n = len(texts_a)
    all_embs = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False,
                            batch_size=2048, convert_to_tensor=True)
    sims = torch.sum(all_embs[:n] * all_embs[n:], dim=1).cpu().numpy()
    return sims.tolist()


def word_count(text):
    return len(text.split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stolen_normal", required=True)
    p.add_argument("--reference_normal", required=True)
    p.add_argument("--stolen_fp", required=True)
    p.add_argument("--reference_fp", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sem_pct", type=float, default=1.0,
                   help="Flag if semantic_similarity < benign p<this>")
    p.add_argument("--len_pct", type=float, default=3.0,
                   help="Flag if word_count < benign p<this>")
    args = p.parse_args()

    with open(args.stolen_normal) as f:
        stolen_normal = json.load(f)["samples"]
    with open(args.reference_normal) as f:
        ref_normal = json.load(f)["samples"]
    with open(args.stolen_fp) as f:
        stolen_fp = json.load(f)
    with open(args.reference_fp) as f:
        ref_fp = json.load(f)

    n_benign = min(len(stolen_normal), len(ref_normal))
    stolen_normal = stolen_normal[:n_benign]
    ref_normal = ref_normal[:n_benign]
    n_fp = len(stolen_fp)

    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=args.device)

    benign_sem = compute_semantic_similarity_batch(
        [s["output"] for s in stolen_normal], [s["output"] for s in ref_normal], embedder)
    benign_len = [word_count(s["output"]) for s in stolen_normal]

    fp_sem = compute_semantic_similarity_batch(
        [s["response"] for s in stolen_fp], [s["response"] for s in ref_fp], embedder)
    fp_len = [word_count(s["response"]) for s in stolen_fp]

    sem_th = float(np.percentile(benign_sem, args.sem_pct))
    len_th = float(np.percentile(benign_len, args.len_pct))

    def flag(sem, length):
        return sem < sem_th or length < len_th

    fpr = 100.0 * sum(flag(benign_sem[i], benign_len[i]) for i in range(n_benign)) / n_benign

    # Restrict to samples where the attack fired (PLA-style methods set `hit`).
    has_hit = any("hit" in s for s in stolen_fp)
    eval_indices = [i for i in range(n_fp) if stolen_fp[i].get("hit", False)] if has_hit \
                   else list(range(n_fp))
    n_eval = len(eval_indices)

    detected = 0
    sig_counts = {"sem": 0, "length": 0}
    for i in eval_indices:
        s, l = fp_sem[i], fp_len[i]
        fired = False
        if s < sem_th:
            sig_counts["sem"] += 1; fired = True
        if l < len_th:
            sig_counts["length"] += 1; fired = True
        if fired:
            detected += 1

    print(f"Benign: {n_benign} | Fingerprint: {n_fp} (eval: {n_eval})")
    print(f"Thresholds: sem<p{args.sem_pct}={sem_th:.3f}  len<p{args.len_pct}={len_th:.0f} words")
    print(f"Benign FPR: {fpr:.2f}%")
    print(f"Detection rate: {detected}/{n_eval} ({100*detected/n_eval:.1f}%)")
    print(f"  per-signal: sem={sig_counts['sem']}  length={sig_counts['length']}")


if __name__ == "__main__":
    main()
