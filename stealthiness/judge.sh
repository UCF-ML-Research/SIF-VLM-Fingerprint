#!/usr/bin/env bash
set -e

export OPENAI_API_KEY=""

cd "$(dirname "$0")/.."

# python3 stealthiness/judge.py \
#   --mode tmr \
#   --report pla/fingerprint/tmr_report_qwen-2.5-7b-instruct.json \
#   --base-dir pla \
#   --output stealthiness/secret_judge_tmr_qwen-2.5-7b.csv

python3 stealthiness/judge.py \
  --mode sif \
  --report sif/sif_report_qwen-2.5-7b-instruct.json \
  --base-dir sif \
  --output stealthiness/secret_judge_sif_qwen-2.5-7b.csv

