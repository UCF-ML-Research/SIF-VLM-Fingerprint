#!/bin/bash
# Shared helpers for verification scripts

PROMPT="Describe the image in detail."
GAMMA=0.5
DELTA=4.0
SEEDING="simple_0"
DTYPE="bf16"
SEED=1234
CLEANUP_HF_CACHE=${CLEANUP_HF_CACHE:-1}

cleanup_model() {
    [ "$CLEANUP_HF_CACHE" != "1" ] && return
    python -c "
from huggingface_hub import scan_cache_dir
cache = scan_cache_dir()
hashes = set()
for repo in cache.repos:
    if repo.repo_id == '$1':
        for rev in repo.revisions:
            hashes.add(rev.commit_hash)
if hashes:
    strategy = cache.delete_revisions(*hashes)
    strategy.execute()
    print(f'  Freed {strategy.expected_freed_size / 1e9:.1f} GB for $1')
" 2>/dev/null || true
}

run_model() {
    local label="$1" model="$2" out_file="$3"
    shift 3
    [ -f "$RESULT_DIR/$out_file" ] && echo "[$label] SKIP: $out_file" && return
    echo "=== [$label] $model ==="
    CUDA_VISIBLE_DEVICES=0 python detect.py \
        --model_name "$model" \
        --in_dir "$FINGERPRINT_DIR" \
        --out_file "$RESULT_DIR/$out_file" \
        --prompt "$PROMPT" \
        --gamma $GAMMA --delta $DELTA \
        --seeding_scheme "$SEEDING" \
        --dtype $DTYPE --seed $SEED "$@"
    echo "[$label] Done -> $out_file"
    cleanup_model "$model"
    echo ""
}
