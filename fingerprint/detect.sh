#!/bin/bash
# Usage: bash detect.sh {pla|ordinary|rna|cropa|difgsm|proflingo|instruction_fingerprint} {llava|qwen}
set -e
cd "$(cd "$(dirname "$0")" && pwd)"

if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx sif; then
    eval "$(conda shell.bash hook)"; conda activate sif
fi

METHOD="$1"
TARGET="$2"
case "$TARGET" in
    llava) MODEL="llava-hf/llava-1.5-7b-hf" ;;
    qwen)  MODEL="Qwen/Qwen2.5-VL-7B-Instruct" ;;
    *) echo "Usage: bash detect.sh {pla|ordinary|rna|cropa|difgsm|proflingo|instruction_fingerprint} {llava|qwen}"; exit 1 ;;
esac

RESULT_DIR="./${TARGET}_verification_results"
mkdir -p "$RESULT_DIR"
OUT_JSON="$RESULT_DIR/source_${METHOD}_${TARGET}.json"

case "$METHOD" in
proflingo)
    SUFFIXES="./fingerprints/proflingo_fingerprint/${TARGET}_proflingo/proflingo_results.json"
    CUDA_VISIBLE_DEVICES=0 python detect.py \
        --mode proflingo --suffixes "$SUFFIXES" \
        --models "$MODEL" --out_json "$OUT_JSON" --dtype bf16
    ;;
instruction_fingerprint)
    ADAPTER="./fingerprints/instruction_fingerprint/${TARGET}_if"
    CUDA_VISIBLE_DEVICES=0 python detect.py \
        --mode instruction_fingerprint --adapter_path "$ADAPTER" \
        --models "$MODEL" --out_json "$OUT_JSON" --dtype bf16
    ;;
pla|ordinary|rna|cropa|difgsm)
    FINGERPRINT_DIR="./fingerprints/${METHOD}_fingerprint/${TARGET}_${METHOD}"
    CUDA_VISIBLE_DEVICES=0 python detect.py \
        --mode image --out_root "$FINGERPRINT_DIR" \
        --models "$MODEL" --out_json "$OUT_JSON" --dtype bf16
    ;;
*)
    echo "Usage: bash detect.sh {pla|ordinary|rna|cropa|difgsm|proflingo|instruction_fingerprint} {llava|qwen}"
    exit 1
    ;;
esac
