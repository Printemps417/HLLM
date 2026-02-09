#!/usr/bin/env bash
set -euo pipefail

BASE_PATH=${1:-/root/autodl-tmp}
OUTPUT_DIR=${2:-"${BASE_PATH}/HLLM/emb_output"}

python3 main.py \
  --config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
  --dataset Pixel200K \
  --checkpoint_dir "${BASE_PATH}/HLLM/saved_model/HLLM-0.pth/HLLM-0.pth" \
  --item_pretrain_dir "${BASE_PATH}/HLLM/pretrained_models/Qwen3-0.6B-FP8" \
  --user_pretrain_dir "${BASE_PATH}/HLLM/pretrained_models/Qwen3-0.6B-FP8" \
  --text_path "${BASE_PATH}/HLLM/information" \
  --val_only True \
  --save_emb_json True \
  --emb_output_dir "${OUTPUT_DIR}"
