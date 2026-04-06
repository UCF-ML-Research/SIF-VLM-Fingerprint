#!/usr/bin/env python3
# Usage: python analyze.py {llava|qwen}

import os, json, re, sys, argparse
import numpy as np
from scipy.stats import norm, binom, chi2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIGS = {
    "llava": {
        "original": "LLaVA-1.5-7B",
        "result_dir": os.path.join(SCRIPT_DIR, "llava_verification_results"),
        "unrelated": {
            "Qwen2.5-VL-7B":  "unrelated_qwen2_5_vl_7b.json",
            "Qwen2.5-VL-3B":  "unrelated_qwen2_5_vl_3b.json",
            "InternVL3-8B":    "unrelated_internvl3_8b.json",
            "LLaVA-1.5-13B":   "unrelated_llava_1_5_13b.json",
        },
        "suspect": {
            "LLaVA-1.5-7B (original)": "suspect_llava_1_5_7b.json",
            "LLaVA-1.5-7B (4-bit)":    "suspect_llava_1_5_7b_4bit.json",
            "LLaVA-1.5-7B (8-bit)":    "suspect_llava_1_5_7b_8bit.json",
            "vsft-LLaVA":               "suspect_vsft_llava.json",
            "edbeeching-vsft":          "suspect_edbeeching_vsft.json",
            "Table-LLaVA":              "suspect_table_llava.json",
        },
    },
    "qwen": {
        "original": "Qwen2.5-VL-7B",
        "result_dir": os.path.join(SCRIPT_DIR, "qwen_verification_results"),
        "unrelated": {
            "LLaVA-1.5-7B":  "unrelated_llava_1_5_7b.json",
            "LLaVA-1.5-13B": "unrelated_llava_1_5_13b.json",
            "InternVL3-8B":   "unrelated_internvl3_8b.json",
            "Qwen2.5-VL-3B":  "unrelated_qwen2_5_vl_3b.json",
        },
        "suspect": {
            "Qwen2.5-VL-7B (original)": "suspect_qwen2_5_vl_7b.json",
            "Qwen2.5-VL-7B (4-bit)":    "suspect_qwen2_5_vl_7b_4bit.json",
            "Qwen2.5-VL-7B (8-bit)":    "suspect_qwen2_5_vl_7b_8bit.json",
            "GUI-Actor-7B":              "suspect_gui_actor_7b.json",
            "ARC-AGI-7B":                "suspect_arc_agi_7b.json",
            "Cosmos-Reason1-7B":         "suspect_cosmos_reason1_7b.json",
            "RolmOCR":                   "suspect_rolmocr.json",
        },
    },
}


def extract_img_id(sample):
    m = re.search(r"img\d{4}", sample.get("image_path", ""))
    return m.group() if m else None


def extract_stats(sample):
    det = sample.get("detect", {})
    if not isinstance(det, dict):
        return None
    z = det.get("z_score")
    if isinstance(z, (list, tuple)):
        z = max(z) if z else None
    if z is None:
        return None
    return {
        "z_score": float(z),
        "green_fraction": float(det.get("green_fraction", 0)),
        "num_green_tokens": int(det.get("num_green_tokens", 0)),
        "num_tokens_scored": int(det.get("num_tokens_scored", 0)),
    }


def load_model_dict(result_dir, filename):
    path = os.path.join(result_dir, filename)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    d = {}
    for s in data.get("samples", []):
        img_id = extract_img_id(s)
        stats = extract_stats(s)
        if img_id and stats:
            d[img_id] = stats
    return d


def adaptive_threshold_test(suspect_dict, unrelated_dicts, common_ids):
    selected = []
    for img_id in sorted(common_ids):
        max_z = max(ud[img_id]["z_score"] for ud in unrelated_dicts)
        max_gf = max(ud[img_id]["green_fraction"] for ud in unrelated_dicts)
        susp = suspect_dict[img_id]
        if susp["z_score"] > max_z:
            k, n = susp["num_green_tokens"], susp["num_tokens_scored"]
            selected.append({"k": k, "n": n, "p_ref": max_gf,
                             "individual_p": binom.sf(k - 1, n, max_gf) if n > 0 else 1.0})

    n_total = len(common_ids)
    if not selected:
        return {"n_sel": 0, "n_total": n_total, "sel_rate": 0.0,
                "z_stat": 0.0, "p_value": 1.0, "fisher_p": 1.0}

    total_green = sum(s["k"] for s in selected)
    expected = sum(s["n"] * s["p_ref"] for s in selected)
    var = sum(s["n"] * s["p_ref"] * (1 - s["p_ref"]) for s in selected)
    z_stat = (total_green - expected) / np.sqrt(var) if var > 0 else 0
    p_value = 1 - norm.cdf(z_stat)
    fisher_stat = -2 * sum(np.log(max(s["individual_p"], 1e-300)) for s in selected)
    fisher_p = chi2.sf(fisher_stat, 2 * len(selected))

    return {"n_sel": len(selected), "n_total": n_total,
            "sel_rate": len(selected) / n_total,
            "z_stat": z_stat, "p_value": p_value, "fisher_p": fisher_p}


def fmt_p(p):
    if p < 1e-15: return "< 1e-15"
    if p < 1e-10: return f"{p:.1e}"
    return f"{p:.2e}"


def evidence(p):
    if p < 0.05:  return "SUSPECT"
    return "UNRELATED"


def print_row(name, r):
    rate = f"{100 * r['sel_rate']:.1f}%"
    z = f"{r['z_stat']:.2f}" if r["n_sel"] > 0 else "N/A"
    print(f"{name:<35} {rate:>8} {z:>8}  {evidence(r['p_value'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=["llava", "qwen"])
    parser.add_argument("--result_dir", default=None)
    args = parser.parse_args()

    cfg = CONFIGS[args.target]
    result_dir = args.result_dir or cfg["result_dir"]

    if not os.path.isdir(result_dir):
        print(f"Not found: {result_dir}"); sys.exit(1)

    unrelated_dicts, unrelated_names = [], []
    for name, fname in cfg["unrelated"].items():
        d = load_model_dict(result_dir, fname)
        if d is None:
            print(f"  MISSING: {fname}"); continue
        unrelated_dicts.append(d); unrelated_names.append(name)
    if len(unrelated_dicts) < 2:
        print("Need >= 2 unrelated models."); sys.exit(1)

    suspect_dicts = {}
    for name, fname in cfg["suspect"].items():
        d = load_model_dict(result_dir, fname)
        if d is None:
            print(f"  MISSING: {fname}"); continue
        suspect_dicts[name] = d
    if not suspect_dicts:
        print("No suspect results found."); sys.exit(1)

    base_ids = set.intersection(*(set(d.keys()) for d in unrelated_dicts))
    print(f"Original model: {cfg['original']}")
    print()
    print(f"{'Model':<35} {'Rate':>8} {'z-stat':>8}  {'Verdict'}")
    print("-" * 65)

    for name, d in suspect_dicts.items():
        ids = base_ids & set(d.keys())
        print_row(name, adaptive_threshold_test(d, unrelated_dicts, ids))

    print("-" * 65)
    for i, name in enumerate(unrelated_names):
        print_row(name, adaptive_threshold_test(unrelated_dicts[i], unrelated_dicts, base_ids))


if __name__ == "__main__":
    main()
