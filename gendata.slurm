#!/bin/bash

#SBATCH --output=logs/%x-%j_%A_%a.out
#SBATCH --error=logs/%x-%j_%A_%a.err
#SBATCH --mail-type=ALL

#SBATCH --job-name=gendata
#SBATCH --time=48:00:00
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100-80g
#SBATCH --cpus-per-gpu=4

module purge
module load miniconda
conda activate circuit

# Aggressive CUDA memory settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1

# Reduce other memory usage
export HF_HUB_CACHE=/tmp/hf_cache_$$
export TRANSFORMERS_CACHE=/tmp/transformers_cache_$$

python gendata.py
