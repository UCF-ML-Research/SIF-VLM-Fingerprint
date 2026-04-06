#!/bin/bash
# Usage: bash generate.sh {llava|qwen}
set -e
cd "$(cd "$(dirname "$0")" && pwd)"

PROMPT="Describe the image in detail."
STEPS=1000
EPS="16/255"
ALPHA="1/255"
SEED=42
DTYPE="bf16"
GAMMA=0.5
DELTA=5.0
SEEDING="simple_0"
Z_THRESHOLD=2.0
START=1
END=50

case "$1" in
llava)
    MODEL="llava-hf/llava-1.5-7b-hf"
    INPUT_DIR="../imagenet/images_llava"
    OUT_DIR="./sif_fingerprint/llava_fingerprint"
    ;;
qwen)
    MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
    INPUT_DIR="../imagenet/images_qwen"
    OUT_DIR="./sif_fingerprint/qwen_fingerprint"
    ;;
*)
    echo "Usage: bash generate.sh {llava|qwen}"
    exit 1
    ;;
esac

CUDA_VISIBLE_DEVICES=0 python sif.py \
  --model_name "$MODEL" \
  --input_dir "$INPUT_DIR" \
  --out_dir "$OUT_DIR" \
  --prompt "$PROMPT" \
  --start $START --end $END \
  --steps $STEPS --eps $EPS --alpha $ALPHA \
  --seed $SEED --dtype $DTYPE \
  --gamma $GAMMA --delta $DELTA \
  --seeding_scheme $SEEDING --z_threshold $Z_THRESHOLD
