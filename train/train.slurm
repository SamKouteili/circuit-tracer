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

# Debug: Show what's available
echo "=== DEBUG INFO ==="
echo "CONDA_EXE: $CONDA_EXE"
echo "Available conda environments:"
conda env list

# Try to activate circuit environment
echo "Activating circuit environment..."
conda activate circuit

# Debug: Check if activation worked
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Test torch import
echo "Testing torch import..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print('torch import successful')" || echo "torch import failed"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "=== STARTING TRAINING ==="
# Run training with the large dataset - small model, reasonable resources
python train.py --dataset samkouteili/injection-attribution-graphs --epochs 100 --batch_size 16 --lr 0.001
