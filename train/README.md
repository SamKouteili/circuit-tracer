# Prompt Injection Detection Training Pipeline

This directory contains a complete training pipeline for detecting prompt injection attempts using Graph Neural Networks (GraphGPS) on attribution graphs.

## Overview

The pipeline loads attribution graphs from HuggingFace datasets, converts them to PyTorch Geometric format, and trains a GraphGPS model to classify graphs as either benign conversation or prompt injection attempts.

## Core Pipeline Files

1. **`data_converter.py`** - Converts JSON attribution graphs to PyTorch Geometric format
2. **`dataset.py`** - Dataset classes and HuggingFace data loading utilities 
3. **`models.py`** - GraphGPS model implementation for prompt injection detection
4. **`convert_and_load_dataset.py`** - HuggingFace dataset downloader and converter
5. **`test_pipeline.py`** - End-to-end testing of the entire pipeline

## Important Context for Future Claude Instances

### HuggingFace Dataset Loading Issue & Solution

**Problem**: HuggingFace datasets with attribution graphs have variable-length arrays (different numbers of nodes/links per graph) which breaks PyArrow's schema inference, causing JSON parse errors.

**Root Cause**: PyArrow expects consistent array lengths across all JSON files, but attribution graphs naturally vary in size.

**Solution**: Local conversion pipeline that bypasses HuggingFace's problematic JSON loader:
1. Downloads raw JSON files from HuggingFace using `huggingface_hub`
2. Converts them locally to expected format: `{"json": "stringified_attribution_graph"}`
3. Loads using our custom PyTorch Geometric converter

**Key Implementation**: `dataset.py` uses `create_datasets_from_huggingface()` which calls `download_and_convert_dataset()` - no fallbacks, single clean path.

## Quick Start

### 1. Test the Pipeline

```bash
cd /Users/samkouteili/rose/circuits/circuit-tracer
python train/test_pipeline.py
```

### 2. Load HuggingFace Dataset

```python
from train.dataset import create_datasets_from_huggingface
from train.models import PromptInjectionGraphGPS

# Load and convert dataset automatically
train_dataset, val_dataset, test_dataset, converter = create_datasets_from_huggingface(
    dataset_name="samkouteili/injection-attribution-graphs",
    test_size=0.2,
    val_size=0.1
)

# Create model
model = PromptInjectionGraphGPS(
    input_dim=converter.get_feature_dim(),
    hidden_dim=256,
    num_layers=6,
    num_heads=8,
    num_classes=2
)
```

## Data Format

### HuggingFace Dataset Structure

The pipeline expects datasets with two subdirectories:
- `benign/` - Contains normal conversation attribution graphs
- `injected/` - Contains prompt injection attribution graphs

Each JSON file contains raw attribution graph data with metadata, nodes, and links.

### Converted Format

After local conversion, files have the expected format:
```json
{
  "json": "{\"nodes\": [...], \"links\": [...]}"
}
```

### PyTorch Geometric Output

Converted to `torch_geometric.data.Data` objects with:
- **Node features** (9 dimensions): influence, activation, layer, ctx_idx, feature_id, + one-hot feature types
- **Edge connections**: Source-target relationships with weights  
- **Graph labels**: 0 = benign, 1 = injected

## Model Architecture

### GraphGPS Layers

The model combines:
1. **Local Message Passing** (GCN) - Nodes communicate with direct neighbors
2. **Global Attention** - All nodes can attend to each other within a graph
3. **Positional Encoding** - Structural position information
4. **Feed-Forward Networks** - Non-linear transformations

### Key Features

- **Linear complexity** O(N+E) instead of O(N²) 
- **Multi-head attention** for rich representations
- **Residual connections** for stable training
- **Layer normalization** for improved convergence
- **Multi-scale readout** (mean + max + sum pooling)

## Directory Structure

```
train/
├── __init__.py
├── README.md                    # This file
├── data_converter.py           # JSON → PyTorch Geometric conversion
├── dataset.py                  # Main dataset loading utilities
├── models.py                   # GraphGPS model implementation
├── convert_and_load_dataset.py # HuggingFace download & conversion
├── test_pipeline.py           # End-to-end pipeline testing
└── analyze/                   # Analysis and debugging files
    ├── analyze_schema.py      # Schema inconsistency analysis
    ├── check_large_dataset.py # Large dataset structure checker
    ├── dataset_backup.py      # Backup of old dataset implementation
    ├── debug_dataset.py       # Dataset debugging utilities
    ├── deep_analyze.py        # Deep field type analysis
    ├── fix_dataset_loading.py # Alternative loading approaches
    ├── test_hf_dataset.py     # HuggingFace loading tests
    └── converted_datasets/    # Cached converted datasets
```

## Available Datasets

- `samkouteili/injection-attribution-graphs` - Full dataset (large)
- `samkouteili/injection-attribution-graphs-small` - Test dataset (20 graphs)

## Current Status

### ✅ Completed Features
- **HuggingFace Integration**: Automatic dataset download and conversion
- **Robust Data Processing**: Handles variable-length attribution graphs  
- **PyTorch Geometric Pipeline**: Full conversion to graph neural network format
- **GraphGPS Model**: Complete implementation with multi-head attention
- **Stratified Splitting**: Proper train/val/test splits maintaining class balance
- **Comprehensive Testing**: End-to-end pipeline validation

### 🔄 Next Steps
1. **Training Loop**: Implement trainer class with metrics and checkpointing
2. **Hyperparameter Tuning**: Experiment with model architectures
3. **Evaluation Metrics**: Add precision, recall, F1-score, AUC
4. **Explainability**: Integrate GNNExplainer for pattern discovery

## Technical Notes

### Environment Setup
- Requires conda environment with `torch`, `torch_geometric`, `datasets`, `huggingface_hub`
- Tested with Python 3.12 and PyTorch 2.7

### Performance
- Small dataset (20 graphs): ~24K unique nodes, 9D features
- Caching: Converted datasets are cached locally to avoid re-download
- Memory: Uses stratified batching for large graph processing

## Dependencies

```python
torch>=2.0
torch_geometric
scikit-learn
numpy
datasets
huggingface_hub
```

## Important Implementation Details

### Why Local Conversion?
- **PyArrow Schema Issue**: HuggingFace's automatic JSON loading fails with variable-length arrays
- **Deterministic Processing**: Local conversion ensures consistent handling across environments
- **Caching**: Avoids re-downloading large datasets repeatedly

### Dataset Format Requirements
- Must have `benign/` and `injected/` subdirectories in HuggingFace dataset
- JSON files can have any internal structure (metadata, qParams, etc.)
- Conversion extracts only `nodes` and `links` arrays for training

This pipeline provides a robust, production-ready foundation for training Graph Neural Networks on attribution graphs to detect prompt injection attempts, with full handling of real-world dataset complexities.