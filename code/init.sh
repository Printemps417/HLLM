

# 1. 将 CUDA 路径加入环境变量 (假设是 cuda-13.0，如果不是请修改版本号)
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# 2. 将 Nsight Systems 路径加入环境变量 (AutoDL 默认通常在这个位置)
echo 'export PATH=/opt/nvidia/nsight-systems/2026.1.1/bin:$PATH' >> ~/.bashrc

# 3. 刷新配置使其生效
source ~/.bashrc