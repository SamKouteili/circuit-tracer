#!/bin/bash
#SBATCH --job-name=explain_graphgps
#SBATCH --output=logs/explain_%j.out
#SBATCH --error=logs/explain_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load required modules
module purge
module load miniconda
module load CUDA/12.6.0
# Activate conda environment
conda activate circuit

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p explanations

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Starting time: $(date)"
echo "Working directory: $(pwd)"

# Default arguments - can be overridden via command line
MODEL_PATH=${1:-"../train/train_output/best_model.pt"}
CACHE_DIR=${2:-"../train/train_cache"}
OUTPUT_DIR=${3:-"explanations/batch_$(date +%Y%m%d_%H%M%S)"}
SPLIT=${4:-"test"}
MAX_GRAPHS=${5:-"500"}
EPOCHS=${6:-"200"}

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Cache Directory: $CACHE_DIR"
echo "  Dataset Split: $SPLIT"
echo "  Output: $OUTPUT_DIR"
echo "  Max Graphs: $MAX_GRAPHS"
echo "  Explanation Epochs: $EPOCHS"
echo ""

# Verify model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    echo "Available model files:"
    find . -name "*.pt" -o -name "*.pth" | head -10
    exit 1
fi

# Verify cache directory exists
if [ ! -d "$CACHE_DIR" ]; then
    echo "Error: Cache directory not found: $CACHE_DIR"
    echo "Available cache directories:"
    find . -maxdepth 2 -type d -name "*cache*" | head -5
    echo ""
    echo "Make sure you have run training first to generate cache files:"
    echo "   python train.py --cache_dir $CACHE_DIR"
    exit 1
fi

# Check for PyG dataset cache files
echo "Checking for PyG dataset cache files..."
CACHE_FILES=$(find "$CACHE_DIR" -name "*_dataset_*.pkl" | wc -l)
if [ "$CACHE_FILES" -eq 0 ]; then
    echo "Error: No PyG dataset cache files found in $CACHE_DIR"
    echo "Expected files like: train_dataset_*.pkl, val_dataset_*.pkl, test_dataset_*.pkl"
    echo ""
    echo "Run training first to generate cache files:"
    echo "   python train.py --cache_dir $CACHE_DIR"
    exit 1
fi

echo "Found $CACHE_FILES PyG dataset cache files:"
find "$CACHE_DIR" -name "*_dataset_*.pkl" -exec basename {} \; | head -3

echo "Files verified. Starting explanation generation..."
echo ""

# Run explanation generation
python explainer/explain_model.py \
    --model "$MODEL_PATH" \
    --cache-dir "$CACHE_DIR" \
    --split "$SPLIT" \
    --mode batch \
    --output-dir "$OUTPUT_DIR" \
    --max-graphs "$MAX_GRAPHS" \
    --epochs "$EPOCHS" \
    --device cuda \
    --use-cache \
    --batch-size 16 \
    --verbose

EXIT_CODE=$?

echo ""
echo "Explanation generation completed with exit code: $EXIT_CODE"
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Success! Results saved to: $OUTPUT_DIR"
    
    # Print summary of generated files
    echo ""
    echo "Generated files:"
    find "$OUTPUT_DIR" -type f -name "*.html" -o -name "*.json" -o -name "*.csv" | head -10
    
    # Print size of output directory
    OUTPUT_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
    echo "Total output size: $OUTPUT_SIZE"
    
    # Check for HTML report
    if [ -f "$OUTPUT_DIR/batch_analysis_report.html" ]; then
        echo ""
        echo "Main report available at: $OUTPUT_DIR/batch_analysis_report.html"
        echo "   To view: scp this file to your local machine and open in browser"
    fi
    
else
    echo "Explanation generation failed"
    echo "Check the error log: logs/explain_${SLURM_JOB_ID}.err"
fi

exit $EXIT_CODE
