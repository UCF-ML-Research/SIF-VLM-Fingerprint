#!/bin/bash
set -e
cd "$(cd "$(dirname "$0")" && pwd)"

CU_INDEX="--extra-index-url https://download.pytorch.org/whl/cu121"
QWEN_OVERRIDE="transformers==4.57.2 tokenizers==0.22.2 accelerate==1.13.0"

mk_env() {
    local name=$1
    echo "===== Creating env: $name ====="
    conda create -y -n "$name" python=3.11
    eval "$(conda shell.bash hook)"
    conda activate "$name"
    pip install -r requirements.txt $CU_INDEX
    pip install -q git+https://github.com/deepseek-ai/DeepSeek-VL.git
    if [ "$name" = "sif-qwen" ]; then
        pip install -U $QWEN_OVERRIDE
    fi
    conda deactivate
    echo "===== Done: $name ====="
}

case "${1:-both}" in
    sif)       mk_env sif ;;
    sif-qwen)  mk_env sif-qwen ;;
    both|"")   mk_env sif; mk_env sif-qwen ;;
    *)         echo "Usage: bash install.sh [sif|sif-qwen|both]"; exit 1 ;;
esac
