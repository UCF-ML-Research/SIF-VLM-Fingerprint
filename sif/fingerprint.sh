python detect.py \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --in_dir ./sif_fingerprint \
  --out_file sif_report_qwen-2.5-7b-instruct.json \
  --prompt "Describe the image in detail." \
  --gamma 0.5 \
  --delta 4.0 \
  --seeding_scheme simple_0 \
  --z_threshold 3\
  --dtype bf16

python detect.py \
  --model_name llava-hf/llava-1.5-7b-hf \
  --in_dir ./sif_fingerprint \
  --out_file sif_report_llava-1.5-7b-hf.json \
  --prompt "Describe the image in detail." \
  --gamma 0.5 \
  --delta 4.0 \
  --seeding_scheme simple_0 \
  --z_threshold 3\
  --dtype bf16  