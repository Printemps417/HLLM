from huggingface_hub import snapshot_download

# 模型名称
model_id = "Qwen/Qwen3-1.7B-FP8"
# 本地保存路径
local_dir = "../pretrain_models/Qwen3-1.7B-FP8"

# 开始下载
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)