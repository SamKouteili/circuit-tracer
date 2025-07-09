#!/usr/bin/env python3
"""
Simple GraphGPS training script using existing dataset.py
"""

from dataset import create_datasets_from_local_directory
from torch_geometric.graphgym.utils.device import auto_select_device
from torch_geometric.graphgym.register import train_dict
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.logger import create_logger
from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.config import cfg, load_cfg
import sys
import os
from pathlib import Path

# Add GraphGPS to path
sys.path.insert(0, str(Path(__file__).parent / "GraphGPS"))


def main():
    # Load our datasets using existing code
    train_dataset, val_dataset, test_dataset, converter = create_datasets_from_local_directory(
        "/home/sk2959/palmer_scratch/data",
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        cache_dir="/home/sk2959/project/circuits/circuit-tracer/train/train_cache"
    )

    print(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Node features: {train_dataset[0].x.shape[1]}")
    print(f"Edge features: {train_dataset[0].edge_attr.shape[1]}")

    # Load GraphGPS config
    config_file = "attribution-graphs-gps.yaml"
    load_cfg(config_file)

    # Set device
    auto_select_device()

    # Create model
    model = create_model()

    # Create optimizer and scheduler
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer)

    # Create loggers
    loggers = create_logger()

    # Create loaders - we'll need to patch this to use our datasets
    # For now, let's just use the standard loader
    loaders = create_loader()

    # Train
    train_func = train_dict[cfg.train.mode]
    train_func(loggers, loaders, model, optimizer, scheduler)


if __name__ == "__main__":
    main()
