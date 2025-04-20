#!/bin/bash
#SBATCH --job-name=pp_vllm_mp
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=GPU
#SBATCH --gres=gpu:v100-32:8   # 请求节点上的所有8个32GB V100 GPU
#SBATCH --constraint=v100-32   # 确保是32GB V100节点

# 创建临时目录用于模型缓存
mkdir -p /tmp/ztong/huggingface_cache
export HF_HOME=/tmp/ztong/huggingface_cache

# 加载模块
module load cuda
module load anaconda3

# 激活环境
source activate vllm_env

# 设定日志级别
export VLLM_LOGGING_LEVEL=INFO

# 运行脚本
python pp_test.py
# 30480221