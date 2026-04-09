#!/usr/bin/env python3
"""Compute response divergence between stolen and reference models.
Calibrates thresholds on benign data, reports fingerprint detection rate.

Usage:
  python compute_divergence.py \
    --stolen_normal results/response_divergence/normal/responses_llava.json \
    --reference_normal results/response_divergence/normal/responses_internvl.json \
    --stolen_fp results/response_divergence/pla/llava/responses_stolen_llava.json \
    --reference_fp results/response_divergence/pla/llava/responses_reference_internvl.json
"""
import json, argparse
import numpy as np
import torch
from collections import Counter
from sentence_transformers import SentenceTransformer


def compute_lexical_overlap(text_a, text_b):
    """Smoothed token F1: (2*common + 1) / (|A| + |B| + 1) on word bags."""
    ca = Counter(text_a.lower().split())
    cb = Counter(text_b.lower().split())
    common = sum((ca & cb).values())
    total = sum(ca.values()) + sum(cb.values())
    return (2 * common + 1) / (total + 1)


def compute_semantic_similarity_batch(texts_a, texts_b, model):
    """Batch cosine similarity of sentence embeddings on GPU."""
    all_texts = texts_a + texts_b
    n = len(texts_a)
    all_embs = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False,
                            batch_size=2048, convert_to_tensor=True)
    sims = torch.sum(all_embs[:n] * all_embs[n:], dim=1).cpu().numpy()
    return sims.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stolen_normal", required=True)
    p.add_argument("--reference_normal", required=True)
    p.add_argument("--stolen_fp", required=True)
    p.add_argument("--reference_fp", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # Load data
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

    # Load embedder
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=args.device)

    # Benign metrics
    benign_lex = [compute_lexical_overlap(stolen_normal[i]["output"], ref_normal[i]["output"])
                  for i in range(n_benign)]
    benign_sem = compute_semantic_similarity_batch(
        [s["output"] for s in stolen_normal], [s["output"] for s in ref_normal], embedder)

    # Fingerprint metrics
    fp_lex = [compute_lexical_overlap(stolen_fp[i]["response"], ref_fp[i]["response"])
              for i in range(n_fp)]
    fp_sem = compute_semantic_similarity_batch(
        [s["response"] for s in stolen_fp], [s["response"] for s in ref_fp], embedder)

    # Thresholds at 5% FP
    lex_th = np.percentile(benign_lex, 5)
    sem_th = np.percentile(benign_sem, 5)

    # Evaluate on hit samples only (or all if no hit field)
    has_hit = any("hit" in s for s in stolen_fp)
    if has_hit:
        eval_indices = [i for i in range(n_fp) if stolen_fp[i].get("hit", False)]
    else:
        eval_indices = list(range(n_fp))
    n_eval = len(eval_indices)
    detected = sum(1 for i in eval_indices if fp_lex[i] < lex_th or fp_sem[i] < sem_th)

    # Print results
    print(f"Benign: {n_benign} | Fingerprint: {n_fp} (eval: {n_eval})")
    print(f"Detection rate: {detected}/{n_eval} ({100*detected/n_eval:.1f}%)")


if __name__ == "__main__":
    main()
