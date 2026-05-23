#!/bin/bash
# Usage: bash detect.sh {llava|qwen}
set -e
cd "$(cd "$(dirname "$0")" && pwd)"

TARGET="$1"
case "$TARGET" in
llava)
    TRIGGER_DIR="${TRIGGER_DIR:-./sif_run/llava}"
    RESULT_DIR="${RESULT_DIR:-./sif_run/verify_llava}"
    ENV_NAME=sif
    ;;
qwen)
    TRIGGER_DIR="${TRIGGER_DIR:-./sif_run/qwen}"
    RESULT_DIR="${RESULT_DIR:-./sif_run/verify_qwen}"
    ENV_NAME=sif-qwen
    ;;
*)
    echo "Usage: bash detect.sh {llava|qwen}"
    exit 1
    ;;
esac
mkdir -p "$RESULT_DIR"

# Auto-activate the matching conda env (llava->sif, qwen->sif-qwen)
if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    eval "$(conda shell.bash hook)"; conda activate "$ENV_NAME"
fi

GPU=${GPU:-0}

run_one() {
    local label="$1" model="$2" out="$3"
    shift 3
    if [ -f "$RESULT_DIR/$out" ]; then
        echo "[$label] SKIP: $out (already exists)"
        return
    fi
    echo "[$label] GPU$GPU: $model -> $out"
    env CUDA_VISIBLE_DEVICES=$GPU $EXTRA_ENV python -u detect.py \
        --trigger_dir "$TRIGGER_DIR" \
        --model "$model" \
        --label "$label" \
        --out_json "$RESULT_DIR/$out" \
        --include_clean \
        "$@" 2>&1 | tee "$RESULT_DIR/${out%.json}.log"
    unset EXTRA_ENV
}

if [ "$TARGET" = "llava" ]; then
    run_one "source"           "llava-hf/llava-1.5-7b-hf"                "source_llava_1_5_7b.json"
    run_one "suspect-int8"     "llava-hf/llava-1.5-7b-hf"                "suspect_llava_1_5_7b_int8.json"    --load_8bit
    run_one "suspect-int4"     "llava-hf/llava-1.5-7b-hf"                "suspect_llava_1_5_7b_int4.json"    --load_4bit
    run_one "suspect-unsloth"   "unsloth/llava-1.5-7b-hf"                 "suspect_unsloth_llava.json"
    run_one "suspect-lessw"     "LESW/llava-1.5-7b-hf"                    "suspect_lessw_llava.json"
    run_one "suspect-rakitha"   "rakitha/llava-1.5-7b-hf-ft-mix-vsft"     "suspect_rakitha.json"
    run_one "suspect-bill"      "sel639/llava-1.5-7b-hf-ft-bill-epoch-120" "suspect_bill.json"
    run_one "suspect-baurrustem" "BaurRustem/llava-1.5-7b-hf-ft-mix-vsft"  "suspect_baurrustem.json"
    run_one "suspect-ubitech"   "ubitech-edg/llava-7b-sft"                "suspect_ubitech.json"
    # non-HF derivatives: transplant LM weights into base llava-hf shell
    EXTRA_ENV="LLAVA_LM_TRANSPLANT=1" \
        run_one "suspect-medcxr-f"  "X-iZhang/Med-CXRGen-F"               "suspect_med_cxrgen_f.json"
    EXTRA_ENV="LLAVA_LM_TRANSPLANT=1" \
        run_one "suspect-medcxr-i"  "X-iZhang/Med-CXRGen-I"               "suspect_med_cxrgen_i.json"
    EXTRA_ENV="LLAVA_LM_TRANSPLANT=1" \
        run_one "suspect-quilt"     "wisdomik/Quilt-Llava-v1.5-7b"        "suspect_quilt_llava.json"
    EXTRA_ENV="LLAVA_LM_TRANSPLANT=1" \
        run_one "suspect-ada-llava" "zhuoyanxu/ada-llava-L-v1.5-7b"       "suspect_ada_llava.json"
    EXTRA_ENV="LLAVA_LM_TRANSPLANT=1" \
        run_one "suspect-lisa"      "Senqiao/LISA_Plus_7b"                "suspect_lisa.json"
    run_one "unrelated-q2.5vl-7b" "Qwen/Qwen2.5-VL-7B-Instruct"           "unrelated_qwen2_5_vl_7b.json"
    run_one "unrelated-q2.5vl-3b" "Qwen/Qwen2.5-VL-3B-Instruct"           "unrelated_qwen2_5_vl_3b.json"
    run_one "unrelated-iv3-8b"    "OpenGVLab/InternVL3-8B"                "unrelated_internvl3_8b.json"
    run_one "unrelated-iv3-2b"    "OpenGVLab/InternVL3-2B"                "unrelated_internvl3_2b.json"
    run_one "unrelated-iv25-1b-mpo" "OpenGVLab/InternVL2_5-1B-MPO"        "unrelated_internvl2_5_1b_mpo.json"
    run_one "unrelated-llava-next" "llava-hf/llava-v1.6-mistral-7b-hf"    "unrelated_llava_next.json"
    run_one "unrelated-deepseek-vl" "deepseek-ai/deepseek-vl-7b-chat"     "unrelated_deepseek_vl.json"
    run_one "unrelated-llava-13b" "llava-hf/llava-1.5-13b-hf"             "unrelated_llava_1_5_13b.json"      --load_8bit
else
    run_one "source"           "Qwen/Qwen2.5-VL-7B-Instruct"             "source_qwen2_5_vl_7b.json"
    run_one "suspect-int8"     "Qwen/Qwen2.5-VL-7B-Instruct"             "suspect_qwen2_5_vl_7b_int8.json"   --load_8bit
    run_one "suspect-int4"     "Qwen/Qwen2.5-VL-7B-Instruct"             "suspect_qwen2_5_vl_7b_int4.json"   --load_4bit
    run_one "suspect-arcagi"   "mertaylin/Qwen2.5-VL-7B-Instruct_arc-agi-train-400" "suspect_arc_agi_7b.json"
    run_one "suspect-rolmocr"  "reducto/RolmOCR"                         "suspect_rolmocr_7b.json"
    run_one "suspect-xreasoner" "microsoft/X-Reasoner-7B"                "suspect_xreasoner.json"
    run_one "suspect-wowolf"   "WoWolf/Qwen2_5vl-7b-fm-tuned"            "suspect_wowolf.json"
    run_one "suspect-abliterated" "huihui-ai/Qwen2.5-VL-7B-Instruct-abliterated" "suspect_abliterated.json"
    run_one "suspect-replan"    "TainU/RePlan-Qwen2.5-VL-7B"              "suspect_replan.json"
    run_one "suspect-deepeyes"  "ChenShawn/DeepEyes-7B"                   "suspect_deepeyes.json"
    run_one "suspect-unsloth-q25" "unsloth/Qwen2.5-VL-7B-Instruct"        "suspect_unsloth_q25.json"
    run_one "unrelated-q2-7b"  "Qwen/Qwen2-VL-7B-Instruct"               "unrelated_qwen2_vl_7b.json"
    run_one "unrelated-q2-2b"  "Qwen/Qwen2-VL-2B-Instruct"               "unrelated_qwen2_vl_2b.json"
    run_one "unrelated-iv3-8b" "OpenGVLab/InternVL3-8B"                  "unrelated_internvl3_8b.json"
    run_one "unrelated-iv3.5"  "OpenGVLab/InternVL3_5-1B-Instruct"       "unrelated_internvl3_5_1b.json"
    run_one "unrelated-iv2"    "OpenGVLab/InternVL2-1B"                  "unrelated_internvl2_1b.json"
    run_one "unrelated-iv25-1b-mpo" "OpenGVLab/InternVL2_5-1B-MPO"       "unrelated_internvl2_5_1b_mpo.json"
fi

echo "Done. Results in $RESULT_DIR"
echo
python detect.py --analyze "$RESULT_DIR"
