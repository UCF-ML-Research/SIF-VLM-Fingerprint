python fingerprint.py \
  --out_root fingerprint/llava_pla \
  --models llava-hf/llava-1.5-7b-hf \
  --out_json fingerprint/tmr_report_llava-1.5-7b-hf.json \
  --dtype bf16  

python fingerprint.py \
  --out_root fingerprint/qwen_pla \
  --models Qwen/Qwen2.5-VL-7B-Instruct \
  --out_json fingerprint/tmr_report_qwen-2.5-7b-instruct.json \
  --dtype bf16  
