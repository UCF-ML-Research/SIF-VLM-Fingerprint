#!/bin/bash
# Usage: bash fingerprint.sh {llava|qwen}
set -e
cd "$(cd "$(dirname "$0")" && pwd)"

DTYPE="bf16"
CLEANUP_HF_CACHE=${CLEANUP_HF_CACHE:-1}

run_model() {
    local label="$1" model="$2" out_file="$3"
    shift 3
    [ -f "$RESULT_DIR/$out_file" ] && echo "[$label] SKIP: $out_file" && return
    echo "=== [$label] $model ==="
    CUDA_VISIBLE_DEVICES=0 python detect.py \
        --mode image \
        --out_root "$FINGERPRINT_DIR" \
        --models "$model" \
        --out_json "$RESULT_DIR/$out_file" \
        --dtype $DTYPE "$@"
    echo "[$label] Done -> $out_file"
    [ "$CLEANUP_HF_CACHE" = "1" ] && python -c "
from huggingface_hub import scan_cache_dir
cache = scan_cache_dir()
hashes = set()
for repo in cache.repos:
    if repo.repo_id == '$model':
        for rev in repo.revisions:
            hashes.add(rev.commit_hash)
if hashes:
    strategy = cache.delete_revisions(*hashes)
    strategy.execute()
" 2>/dev/null || true
    echo ""
}

case "$1" in
llava)
    FINGERPRINT_DIR="./fingerprints/pla_fingerprint/llava_pla"
    RESULT_DIR="./llava_verification_results"
    mkdir -p "$RESULT_DIR"

    run_model "1/5" "llava-hf/llava-1.5-7b-hf"                  "suspect_llava_1_5_7b.json"
    run_model "2/5" "llava-hf/llava-1.5-7b-hf"                  "suspect_llava_1_5_7b_4bit.json"  --load_4bit
    run_model "3/5" "llava-hf/llava-1.5-7b-hf"                  "suspect_llava_1_5_7b_8bit.json"  --load_8bit
    run_model "4/5" "HuggingFaceH4/vsft-llava-1.5-7b-hf-trl"   "suspect_vsft_llava.json"
    run_model "5/5" "edbeeching/vsft-llava-1.5-7b-hf"           "suspect_edbeeching_vsft.json"
    ;;
qwen)
    FINGERPRINT_DIR="./fingerprints/pla_fingerprint/qwen_pla"
    RESULT_DIR="./qwen_verification_results"
    mkdir -p "$RESULT_DIR"

    run_model "1/7" "Qwen/Qwen2.5-VL-7B-Instruct"      "suspect_qwen2_5_vl_7b.json"
    run_model "2/7" "Qwen/Qwen2.5-VL-7B-Instruct"      "suspect_qwen2_5_vl_7b_4bit.json"  --load_4bit
    run_model "3/7" "Qwen/Qwen2.5-VL-7B-Instruct"      "suspect_qwen2_5_vl_7b_8bit.json"  --load_8bit
    run_model "4/7" "microsoft/GUI-Actor-7B-Qwen2.5-VL" "suspect_gui_actor_7b.json"
    run_model "5/7" "mertaylin/Qwen2.5-VL-7B-Instruct_arc-agi-train-400" "suspect_arc_agi_7b.json"
    run_model "6/7" "nvidia/Cosmos-Reason1-7B"           "suspect_cosmos_reason1_7b.json"
    run_model "7/7" "reducto/RolmOCR"                    "suspect_rolmocr.json"
    ;;
*)
    echo "Usage: bash fingerprint.sh {llava|qwen}"
    exit 1
    ;;
esac
