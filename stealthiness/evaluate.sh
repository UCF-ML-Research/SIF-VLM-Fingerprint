#!/bin/bash
# Usage: bash evaluate.sh {reference|gpt} {sif|pla|proflingo} {llava|qwen}
set -e
cd "$(cd "$(dirname "$0")" && pwd)"

MODE="$1"
METHOD="$2"
TARGET="$3"
REFERENCE="Qwen/Qwen3.5-0.8B"
GPT_MODEL="gpt-4.1-2025-04-14"
NUM_BENIGN=100
FP_RATE=0.05
DTYPE="bf16"

case "$TARGET" in
llava) STOLEN="llava-hf/llava-1.5-7b-hf" ;;
qwen)  STOLEN="Qwen/Qwen2.5-VL-7B-Instruct" ;;
*)     echo "Usage: bash evaluate.sh {reference|gpt} {sif|pla|proflingo} {llava|qwen}"; exit 1 ;;
esac

case "$METHOD" in
sif)       FDIR="../../sif/sif_fingerprint/${TARGET}_fingerprint" ;;
pla)       FDIR="../../fingerprint/fingerprints/pla_fingerprint/${TARGET}_pla" ;;
proflingo) FDIR="../../fingerprint/fingerprints/proflingo_fingerprint/${TARGET}_proflingo/proflingo_results.json" ;;
*)         echo "Usage: bash evaluate.sh {reference|gpt} {sif|pla|proflingo} {llava|qwen}"; exit 1 ;;
esac

mkdir -p results

if [ "$MODE" = "reference" ]; then
    CUDA_VISIBLE_DEVICES=0 python evaluate.py \
      --mode reference --stolen_model "$STOLEN" --reference_model "$REFERENCE" \
      --method "$METHOD" --fingerprint_dir "$FDIR" \
      --num_benign $NUM_BENIGN --fp_rate $FP_RATE \
      --dtype $DTYPE --out_json "results/sda_${MODE}_${METHOD}_${TARGET}.json"
elif [ "$MODE" = "gpt" ]; then
    python evaluate.py \
      --mode gpt --stolen_model "$STOLEN" --gpt_model "$GPT_MODEL" \
      --method "$METHOD" --fingerprint_dir "$FDIR" \
      --dtype $DTYPE --out_json "results/sda_${MODE}_${METHOD}_${TARGET}.json"
else
    echo "Usage: bash evaluate.sh {reference|gpt} {sif|pla|proflingo} {llava|qwen}"
    exit 1
fi
