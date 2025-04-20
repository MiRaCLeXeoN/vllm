#!/bin/bash
#SBATCH --job-name=srun_vll
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=GPU  # 使用GPU分区
#SBATCH --gres=gpu:v100-16:8  # 请求节点上的所有16GB V100 GPU
#SBATCH --gres=gpu:v100-32:8  # 请求节点上的所有32GB V100 GPU

# Load necessary modules
module load cuda
module load anaconda3

# Activate your conda environment
source activate vllm_env

# Set environment variables
export VLLM_LOGGING_LEVEL=DEBUG
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 使用所有可用的GPU

# Run the Python script
python test.py