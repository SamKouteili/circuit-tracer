#!/bin/bash

#SBATCH --output=logs/%x-%j_%A_%a.out
#SBATCH --error=logs/%x-%j_%A_%a.err
#SBATCH --mail-type=ALL

#SBATCH --job-name=training
#SBATCH --time=8:00:00
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100-40g
#SBATCH --cpus-per-gpu=2

module purge
module load miniconda
conda activate circuit

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Run training with the large dataset - small model, reasonable resources
python train.py --dataset samkouteili/injection-attribution-graphs --epochs 100 --batch_size 16 --lr 0.001
