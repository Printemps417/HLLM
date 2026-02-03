BASE_PATH=${1:-/home/runoob/repo}

python3 main.py \
--config_file overall/LLM_deepspeed.yaml HLLM/HLLM.yaml \
--loss nce \
--epochs 3 \
--train_batch_size 2 \
--MAX_TEXT_LENGTH 128 \
--MAX_ITEM_LIST_LENGTH 10 \
--optim_args.learning_rate 1e-4 \
--dataset Pixel200K \
--checkpoint_dir "${BASE_PATH}/HLLM/saved_model" \
--item_pretrain_dir "${BASE_PATH}/HLLM/pretrained_models/Qwen3-0.6B-FP8" \
--user_pretrain_dir "${BASE_PATH}/HLLM/pretrained_models/Qwen3-0.6B-FP8" \
--text_path "${BASE_PATH}/HLLM/information"
