#!/bin/bash
set -e
cd "$(cd "$(dirname "$0")" && pwd)"

# Auto-activate the right conda env (sif for both targets — LLaVA-1.5 source needs transformers 4.49).
if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx sif; then
    eval "$(conda shell.bash hook)"; conda activate sif
fi

PROMPT="Describe the image in detail."
STEPS=500
EPS="16/255"
ALPHA="1/255"
SEED=42
DTYPE="bf16"
GAMMA=0.5
DELTA=5.0
SEEDING="simple_0"
Z_THRESHOLD=2.0
LAMBDA_WM=1.0
LAMBDA_CE=0.5
TOPK=10
MAX_DISTILL_TOKENS=96
RHO=0.05
RFO_LAYERS="every_8"
EVAL_EVERY=100
MAX_NEW_TOKENS_EVAL=200
EVAL_N_SAMPLES=3

START=1
END=50

case "$1" in
llava)
    MODEL="llava-hf/llava-1.5-7b-hf"
    INPUT_DIR="../imagenet/images_llava"
    OUT_DIR="./sif_run/llava"
    ;;
qwen)
    MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
    INPUT_DIR="../imagenet/images_qwen"
    OUT_DIR="./sif_run/qwen"
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
  --seeding_scheme $SEEDING --z_threshold $Z_THRESHOLD \
  --lambda_wm $LAMBDA_WM --lambda_ce $LAMBDA_CE --topk $TOPK \
  --max_distill_tokens $MAX_DISTILL_TOKENS \
  --rfo --rho $RHO --rfo_layers $RFO_LAYERS \
  --eval_every $EVAL_EVERY --max_new_tokens_eval $MAX_NEW_TOKENS_EVAL \
  --eval_n_samples $EVAL_N_SAMPLES
