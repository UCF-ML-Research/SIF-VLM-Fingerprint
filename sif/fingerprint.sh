#!/bin/bash
# Usage: bash fingerprint.sh {llava|qwen}
set -e
cd "$(cd "$(dirname "$0")" && pwd)"
source verify_common.sh

case "$1" in
llava)
    FINGERPRINT_DIR="./sif_fingerprint/llava_fingerprint"
    RESULT_DIR="./llava_verification_results"
    mkdir -p "$RESULT_DIR"

    run_model "1/12"  "Qwen/Qwen2.5-VL-7B-Instruct"              "unrelated_qwen2_5_vl_7b.json"
    run_model "2/12"  "Qwen/Qwen2.5-VL-3B-Instruct"              "unrelated_qwen2_5_vl_3b.json"
    run_model "3/12"  "OpenGVLab/InternVL3-8B"                    "unrelated_internvl3_8b.json"
    run_model "4/12"  "llava-hf/llava-1.5-13b-hf"                 "unrelated_llava_1_5_13b.json"      --load_8bit
    run_model "5/12"  "llava-hf/llava-1.5-7b-hf"                  "suspect_llava_1_5_7b.json"
    run_model "6/12"  "llava-hf/llava-1.5-7b-hf"                  "suspect_llava_1_5_7b_4bit.json"    --load_4bit
    run_model "7/12"  "llava-hf/llava-1.5-7b-hf"                  "suspect_llava_1_5_7b_8bit.json"    --load_8bit
    run_model "8/12"  "HuggingFaceH4/vsft-llava-1.5-7b-hf-trl"   "suspect_vsft_llava.json"
    run_model "9/12"  "edbeeching/vsft-llava-1.5-7b-hf"           "suspect_edbeeching_vsft.json"
    run_model "10/12" "SpursgoZmy/table-llava-v1.5-7b-hf"         "suspect_table_llava.json"
    python analyze.py llava
    ;;
qwen)
    FINGERPRINT_DIR="./sif_fingerprint/qwen_fingerprint"
    RESULT_DIR="./qwen_verification_results"
    mkdir -p "$RESULT_DIR"

    run_model "1/11"  "llava-hf/llava-1.5-7b-hf"        "unrelated_llava_1_5_7b.json"
    run_model "2/11"  "llava-hf/llava-1.5-13b-hf"       "unrelated_llava_1_5_13b.json"       --load_8bit
    run_model "3/11"  "OpenGVLab/InternVL3-8B"           "unrelated_internvl3_8b.json"
    run_model "4/11"  "Qwen/Qwen2.5-VL-3B-Instruct"     "unrelated_qwen2_5_vl_3b.json"
    run_model "5/11"  "Qwen/Qwen2.5-VL-7B-Instruct"     "suspect_qwen2_5_vl_7b.json"
    run_model "6/11"  "Qwen/Qwen2.5-VL-7B-Instruct"     "suspect_qwen2_5_vl_7b_4bit.json"   --load_4bit
    run_model "7/11"  "Qwen/Qwen2.5-VL-7B-Instruct"     "suspect_qwen2_5_vl_7b_8bit.json"   --load_8bit
    run_model "8/11"  "microsoft/GUI-Actor-7B-Qwen2.5-VL"                       "suspect_gui_actor_7b.json"
    run_model "9/11"  "mertaylin/Qwen2.5-VL-7B-Instruct_arc-agi-train-400"     "suspect_arc_agi_7b.json"
    run_model "10/11" "nvidia/Cosmos-Reason1-7B"          "suspect_cosmos_reason1_7b.json"
    run_model "11/11" "reducto/RolmOCR"                   "suspect_rolmocr.json"
    python analyze.py qwen
    ;;
*)
    echo "Usage: bash fingerprint.sh {llava|qwen}"
    exit 1
    ;;
esac
