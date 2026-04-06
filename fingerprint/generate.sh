#!/bin/bash
# Usage: bash generate.sh {pla|ordinary|rna|cropa|difgsm} {llava|qwen}
#        bash generate.sh proflingo {llava|qwen}
set -e
cd "$(cd "$(dirname "$0")" && pwd)"

METHOD="$1"
TARGET="$2"
SEED=42
DTYPE="bf16"

# Text-only methods (no images needed)
if [ "$METHOD" = "proflingo" ] || [ "$METHOD" = "instruction_fingerprint" ]; then
    case "$TARGET" in
    llava)  MODEL="llava-hf/llava-1.5-7b-hf" ;;
    qwen)   MODEL="Qwen/Qwen2.5-VL-7B-Instruct" ;;
    *)      echo "Usage: bash generate.sh {METHOD} {llava|qwen}"; exit 1 ;;
    esac

    if [ "$METHOD" = "proflingo" ]; then
        CUDA_VISIBLE_DEVICES=0 python generate.py \
          --method proflingo \
          --model_name "$MODEL" \
          --out_dir "./fingerprints/proflingo_fingerprint/${TARGET}_proflingo" \
          --seed $SEED --dtype $DTYPE \
          --num_epoch 64 --token_nums 32 --num_questions 50
    else
        CUDA_VISIBLE_DEVICES=0 python generate.py \
          --method instruction_fingerprint \
          --model_name "$MODEL" \
          --out_dir "./fingerprints/instruction_fingerprint/${TARGET}_if" \
          --seed $SEED --dtype $DTYPE \
          --num_fingerprint 10 --if_epochs 20 --if_lr 1e-2 \
          --if_inner_dim 16 --if_batch_size 1
    fi
    exit 0
fi

# Image-based attacks
STEPS=1000
EPS="16/255"
ALPHA="1/255"
BETA=1e-4
CLIP_TH=5e-3
LAM=1e-4
ALPHA2=0.01
CROPA_END=300
MOMENTUM=1.0
DI_PROB=0.5
DI_RESIZE=30
START=1
END=200

case "$TARGET" in
llava)
    MODEL="llava-hf/llava-1.5-7b-hf"
    MODEL_TYPE="llava"
    INPUT_DIR="../imagenet/images_llava"
    ;;
qwen)
    MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
    MODEL_TYPE="qwen"
    INPUT_DIR="../imagenet/images_qwen"
    ;;
*)
    echo "Usage: bash generate.sh {pla|ordinary|rna|cropa|difgsm|proflingo} {llava|qwen}"
    exit 1
    ;;
esac

case "$METHOD" in
pla)      OUT_DIR="./fingerprints/pla_fingerprint/${TARGET}_pla";      EXTRA="--beta $BETA --clip_th $CLIP_TH" ;;
ordinary) OUT_DIR="./fingerprints/pla_fingerprint/${TARGET}_ordinary";  EXTRA="" ;;
rna)      OUT_DIR="./fingerprints/pla_fingerprint/${TARGET}_rna";      EXTRA="--lam $LAM" ;;
cropa)    OUT_DIR="./fingerprints/pla_fingerprint/${TARGET}_cropa";    EXTRA="--alpha2 $ALPHA2 --cropa_end $CROPA_END" ;;
difgsm)   OUT_DIR="./fingerprints/pla_fingerprint/${TARGET}_difgsm";   EXTRA="--momentum $MOMENTUM --di_prob $DI_PROB --di_resize_range $DI_RESIZE" ;;
*)        echo "Usage: bash generate.sh {pla|ordinary|rna|cropa|difgsm|proflingo} {llava|qwen}"; exit 1 ;;
esac

CUDA_VISIBLE_DEVICES=0 python generate.py \
  --method "$METHOD" \
  --model_name "$MODEL" \
  --model_type "$MODEL_TYPE" \
  --input_dir "$INPUT_DIR" \
  --out_dir "$OUT_DIR" \
  --steps $STEPS --eps $EPS --alpha $ALPHA \
  --seed $SEED --dtype $DTYPE \
  --start $START --end $END \
  $EXTRA
