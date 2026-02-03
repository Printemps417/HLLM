import os
import sys
import torch
import subprocess
from transformers import AutoModelForCausalLM, AutoConfig

# ---------------------- 配置参数（必须修改为你的实际路径！）----------------------
DS_CHECKPOINT_DIR = "/root/autodl-tmp/HLLM/saved_model/HLLM-0.pth/checkpoint" 
CONFIG_FILE_PATH = "/root/autodl-tmp/HLLM/pretrained_models/Qwen3-0.6B-FP8/config.json"
# 3. 最终输出目录（生成 pytorch_model.bin 和 config.json）
OUTPUT_DIR = "/root/autodl-tmp/HLLM/pretrained_models/Qwen3-0.6B-FP8/inferred_model"
# 4. 临时文件目录（合并后的中间 .pt 文件，转换完成后会自动删除）
TEMP_MERGED_FILE = "./temp_merged_model.pt"
# 5. DeepSpeed zero_to_fp32.py 脚本路径（当前目录下就有，无需修改）
ZERO_TO_FP32_SCRIPT = "/root/autodl-tmp/HLLM/saved_model/HLLM-0.pth/zero_to_fp32.py"
# --------------------------------------------------------------------------------

def check_file_exists(file_path):
    """检查文件是否存在，不存在则退出"""
    if not os.path.exists(file_path):
        print(f"❌ 错误：文件/目录 {file_path} 不存在！")
        sys.exit(1)

def merge_deepspeed_checkpoint(ds_checkpoint_dir, output_merged_file, zero_script):
    """第一步：用 DeepSpeed 脚本合并分布式 checkpoint"""
    print("="*50)
    print("✅ 开始合并 DeepSpeed 分布式 checkpoint...")
    
    # 检查必要文件
    check_file_exists(ds_checkpoint_dir)
    check_file_exists(zero_script)
    
    # 构建合并命令（调用 zero_to_fp32.py）
    cmd = [
        sys.executable,  # 使用当前 Python 环境
        zero_script,
        ds_checkpoint_dir,
        output_merged_file
    ]
    
    try:
        # 执行合并命令
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ DeepSpeed checkpoint 合并成功！")
        print(f"✅ 中间合并文件已保存到：{output_merged_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ DeepSpeed checkpoint 合并失败！错误信息：{e.stderr}")
        sys.exit(1)

def convert_merged_pt_to_hf_bin(merged_pt_file, config_path, output_dir):
    """第二步：将合并后的 .pt 文件转成 Hugging Face pytorch_model.bin"""
    print("="*50)
    print("✅ 开始转换为 Hugging Face 标准格式...")
    
    # 检查必要文件
    check_file_exists(merged_pt_file)
    check_file_exists(config_path)
    
    # 1. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. 加载模型配置
    print("✅ 加载模型配置...")
    config = AutoConfig.from_pretrained(config_path)
    
    # 3. 初始化空模型（按配置创建架构）
    print("✅ 初始化空模型架构...")
    model = AutoModelForCausalLM.from_config(config)
    
    # 4. 加载合并后的 .pt 权重（加载到 CPU，避免显存不足）
    print("✅ 加载合并后的 .pt 权重文件...")
    pt_weights = torch.load(merged_pt_file, map_location="cpu")
    
    # 5. 处理权重格式（清理冗余前缀、提取 model_state_dict）
    print("✅ 清理并匹配权重格式...")
    # 提取 model_state_dict（如果有嵌套）
    if "model_state_dict" in pt_weights:
        model_weights = pt_weights["model_state_dict"]
    else:
        model_weights = pt_weights
    
    # 移除 module. 前缀（DeepSpeed 分布式训练可能添加）
    cleaned_weights = {}
    for key, value in model_weights.items():
        new_key = key[len("module."):] if key.startswith("module.") else key
        cleaned_weights[new_key] = value
    
    # 6. 加载权重到模型
    try:
        model.load_state_dict(cleaned_weights, strict=False)  # strict=False 兼容部分非核心权重不匹配
        print("✅ 权重成功加载到模型，格式匹配！")
    except Exception as e:
        print(f"⚠️  权重部分不匹配（非致命，可继续）：{e}")
    
    # 7. 保存为 Hugging Face 标准格式（pytorch_model.bin）
    print("✅ 保存为 pytorch_model.bin...")
    model.save_pretrained(
        output_dir,
        save_config=True,  # 自动复制 config.json 到输出目录
        safe_serialization=False  # 保存为 pytorch_model.bin（True 则保存为 safetensors 格式）
    )
    
    print(f"✅ Hugging Face 格式模型保存完成！目录：{output_dir}")
    return True

def clean_temp_file(temp_file):
    """删除临时合并文件，清理磁盘空间"""
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print("="*50)
        print(f"✅ 已删除临时文件：{temp_file}")

def main():
    """主流程：合并 → 转换 → 清理"""
    try:
        # 第一步：合并 DeepSpeed checkpoint
        merge_success = merge_deepspeed_checkpoint(
            DS_CHECKPOINT_DIR,
            TEMP_MERGED_FILE,
            ZERO_TO_FP32_SCRIPT
        )
        
        if not merge_success:
            sys.exit(1)
        
        # 第二步：转换为 pytorch_model.bin
        convert_success = convert_merged_pt_to_hf_bin(
            TEMP_MERGED_FILE,
            CONFIG_FILE_PATH,
            OUTPUT_DIR
        )
        
        if not convert_success:
            sys.exit(1)
        
    finally:
        # 第三步：清理临时文件（无论成功与否，都删除临时文件）
        clean_temp_file(TEMP_MERGED_FILE)
    
    print("="*50)
    print("🎉 一键转换全部完成！最终文件在：", OUTPUT_DIR)
    print(f"🎉 可通过 AutoModelForCausalLM.from_pretrained('{OUTPUT_DIR}') 加载使用")

if __name__ == "__main__":
    main()