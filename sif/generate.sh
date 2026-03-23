python sif.py \
  --model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --input_dir images_qwen \
  --out_dir ./sif_fingerprint \
  --prompt "Describe the image in detail." \
  --start 1 --end 50 \
  --steps 1000 --eps 16/255 --alpha 1/255 \
  --seed 42 --dtype bf16 --gamma 0.5 --delta 5.0 \
  --seeding_scheme simple_0 --z_threshold 2.0 