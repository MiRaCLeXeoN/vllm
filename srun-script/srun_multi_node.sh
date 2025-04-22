#!/bin/bash
#SBATCH --job-name=srun_vllm_sd
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=Compute
#SBATCH --nodelist=hepnode[2-3]

# Load nvidia toolkit
spack load cuda@12.8.0 cmake@3.27.9 gcc@11.4.0

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1

export VLLM_USE_V1=1
srun ray start --block 
# Run the Python script
srun python sd_test.py