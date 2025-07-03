#!/usr/bin/env python3
"""
Script to validate PyG data conversion and inspect dataset quality
"""

import json
import torch
import numpy as np
from pathlib import Path
import argparse
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from train.dataset import create_datasets_from_local_directory, create_data_loaders
    from train.data_converter import AttributionGraphConverter
except ImportError:
    from dataset import create_datasets_from_local_directory, create_data_loaders
    from data_converter import AttributionGraphConverter


def inspect_raw_graph(json_file_path):
    """Inspect a raw JSON attribution graph"""
    print(f"\n=== Inspecting Raw Graph: {Path(json_file_path).name} ===")
    
    with open(json_file_path, 'r') as f:
        if json_file_path.endswith('.json'):
            graph_data = json.load(f)
        else:
            # Assume it's a converted file with JSON string
            data = json.load(f)
            if 'json' in data:
                graph_data = json.loads(data['json'])
            else:
                graph_data = data
    
    print(f"Graph keys: {list(graph_data.keys())}")
    
    if 'nodes' in graph_data:
        nodes = graph_data['nodes']
        print(f"Total nodes: {len(nodes)}")
        
        # Sample first few nodes
        print("\nFirst 3 nodes:")
        for i, node in enumerate(nodes[:3]):
            print(f"  Node {i}: {dict(node)}")
        
        # Feature type distribution
        feature_types = [node.get('feature_type', 'unknown') for node in nodes]
        type_counts = Counter(feature_types)
        print(f"\nFeature type distribution:")
        for ftype, count in type_counts.most_common():
            print(f"  {ftype}: {count}")
        
        # Check for numeric features
        influences = [node.get('influence', 0) for node in nodes if node.get('influence') is not None]
        activations = [node.get('activation', 0) for node in nodes if node.get('activation') is not None]
        
        if influences:
            print(f"\nInfluence stats: min={min(influences):.4f}, max={max(influences):.4f}, mean={np.mean(influences):.4f}")
        if activations:
            print(f"Activation stats: min={min(activations):.4f}, max={max(activations):.4f}, mean={np.mean(activations):.4f}")
    
    if 'links' in graph_data or 'edges' in graph_data:
        edges_key = 'links' if 'links' in graph_data else 'edges'
        edges = graph_data[edges_key]
        print(f"\nTotal edges: {len(edges)}")
        
        if edges:
            print("First 3 edges:")
            for i, edge in enumerate(edges[:3]):
                print(f"  Edge {i}: {dict(edge)}")
            
            # Edge weight distribution
            weights = [edge.get('weight', edge.get('value', 1.0)) for edge in edges]
            weights = [w for w in weights if w is not None]
            if weights:
                print(f"Edge weights: min={min(weights):.4f}, max={max(weights):.4f}, mean={np.mean(weights):.4f}")


def inspect_pyg_conversion(json_file_path, converter, label):
    """Test conversion of a single file to PyG"""
    print(f"\n=== Testing PyG Conversion: {Path(json_file_path).name} ===")
    
    # Read file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Handle both formats
    if 'json' in data:
        json_string = data['json']
    elif 'nodes' in data and 'links' in data:
        json_string = json.dumps(data)
    else:
        print("❌ Unknown file format")
        return None
    
    # Convert to PyG
    pyg_data = converter.json_string_to_pyg_data(json_string, label=label)
    
    if pyg_data is None:
        print("❌ Conversion failed!")
        return None
    
    print("✅ Conversion successful!")
    print(f"PyG Data structure:")
    print(f"  Nodes: {pyg_data.num_nodes}")
    print(f"  Edges: {pyg_data.edge_index.shape[1]}")
    print(f"  Node features shape: {pyg_data.x.shape}")
    print(f"  Edge attributes shape: {pyg_data.edge_attr.shape}")
    print(f"  Label: {pyg_data.y.item()}")
    
    # Feature statistics
    x = pyg_data.x
    print(f"\nNode feature statistics:")
    print(f"  Shape: {x.shape}")
    print(f"  Min values: {x.min(dim=0)[0]}")
    print(f"  Max values: {x.max(dim=0)[0]}")
    print(f"  Mean values: {x.mean(dim=0)}")
    print(f"  Std values: {x.std(dim=0)}")
    
    # Check for any obvious issues
    has_nan = torch.isnan(x).any()
    has_inf = torch.isinf(x).any()
    all_zeros = (x == 0).all(dim=1).sum()
    
    print(f"\nData quality checks:")
    print(f"  Has NaN: {has_nan}")
    print(f"  Has Inf: {has_inf}")
    print(f"  All-zero nodes: {all_zeros}/{pyg_data.num_nodes}")
    
    return pyg_data


def compare_datasets(train_dataset, val_dataset, test_dataset):
    """Compare dataset statistics between splits"""
    print(f"\n=== Dataset Comparison ===")
    
    def analyze_split(dataset, name):
        if len(dataset) == 0:
            return {}
        
        # Collect statistics
        node_counts = []
        edge_counts = []
        labels = []
        feature_stats = {'min': [], 'max': [], 'mean': []}
        
        for data in dataset:
            node_counts.append(data.num_nodes)
            edge_counts.append(data.edge_index.shape[1])
            labels.append(data.y.item())
            
            x = data.x
            feature_stats['min'].append(x.min(dim=0)[0])
            feature_stats['max'].append(x.max(dim=0)[0])
            feature_stats['mean'].append(x.mean(dim=0))
        
        # Convert to tensors for easier analysis
        for key in feature_stats:
            if feature_stats[key]:
                feature_stats[key] = torch.stack(feature_stats[key])
        
        return {
            'node_counts': node_counts,
            'edge_counts': edge_counts,
            'labels': labels,
            'feature_stats': feature_stats
        }
    
    train_stats = analyze_split(train_dataset, "Train")
    val_stats = analyze_split(val_dataset, "Val") 
    test_stats = analyze_split(test_dataset, "Test")
    
    for split_name, stats in [("Train", train_stats), ("Val", val_stats), ("Test", test_stats)]:
        if not stats:
            continue
            
        print(f"\n{split_name} Split:")
        print(f"  Graphs: {len(stats['labels'])}")
        print(f"  Labels: {Counter(stats['labels'])}")
        print(f"  Nodes per graph: min={min(stats['node_counts'])}, max={max(stats['node_counts'])}, mean={np.mean(stats['node_counts']):.1f}")
        print(f"  Edges per graph: min={min(stats['edge_counts'])}, max={max(stats['edge_counts'])}, mean={np.mean(stats['edge_counts']):.1f}")
        
        if stats['feature_stats']['mean'].numel() > 0:
            global_mean = stats['feature_stats']['mean'].mean(dim=0)
            print(f"  Average feature means: {global_mean}")


def main():
    parser = argparse.ArgumentParser(description='Validate PyG data conversion')
    parser.add_argument('dataset_dir', help='Directory containing benign/ and injected/ subdirectories')
    parser.add_argument('--cache_dir', default='./validation_cache', help='Cache directory')
    parser.add_argument('--sample_files', type=int, default=5, help='Number of sample files to inspect per class')
    parser.add_argument('--skip_dataset_creation', action='store_true', help='Skip full dataset creation (faster)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_dir)
    benign_dir = dataset_path / "benign"
    injected_dir = dataset_path / "injected"
    
    if not benign_dir.exists() or not injected_dir.exists():
        print(f"❌ Directories not found: {benign_dir}, {injected_dir}")
        return
    
    print("🔍 PyG Data Conversion Validation")
    print("=" * 50)
    
    # Get ALL files first, then randomly sample
    import random
    random.seed(args.seed)
    
    all_benign_files = list(benign_dir.glob("*.json"))
    all_injected_files = list(injected_dir.glob("*.json"))
    
    print(f"Found {len(all_benign_files)} benign and {len(all_injected_files)} injected files total")
    
    # Randomly sample files
    benign_files = random.sample(all_benign_files, min(args.sample_files, len(all_benign_files)))
    injected_files = random.sample(all_injected_files, min(args.sample_files, len(all_injected_files)))
    
    print(f"Randomly sampled {len(benign_files)} benign and {len(injected_files)} injected files for validation")
    
    # Initialize converter
    converter = AttributionGraphConverter()
    
    # Test vocabulary building with sample files
    print(f"\n1️⃣ Testing vocabulary building...")
    all_sample_files = [str(f) for f in benign_files + injected_files]
    vocab_success = converter.build_vocabulary(all_sample_files)
    
    if vocab_success:
        print(f"✅ Vocabulary built: {len(converter.node_vocab)} unique nodes")
        print(f"Feature dimension: {converter.get_feature_dim()}")
    else:
        print("❌ Vocabulary building failed!")
        return
    
    # Inspect raw graphs
    print(f"\n2️⃣ Inspecting raw graphs...")
    for i, file_path in enumerate(benign_files[:2]):
        inspect_raw_graph(file_path)
    for i, file_path in enumerate(injected_files[:2]):
        inspect_raw_graph(file_path)
    
    # Test PyG conversion
    print(f"\n3️⃣ Testing PyG conversion...")
    for i, file_path in enumerate(benign_files):
        pyg_data = inspect_pyg_conversion(file_path, converter, label=0)
        if i >= 1:  # Limit output
            break
    
    for i, file_path in enumerate(injected_files):
        pyg_data = inspect_pyg_conversion(file_path, converter, label=1)
        if i >= 1:  # Limit output
            break
    
    # Full dataset analysis (optional)
    if not args.skip_dataset_creation:
        print(f"\n4️⃣ Creating and analyzing full datasets...")
        try:
            train_dataset, val_dataset, test_dataset, converter = create_datasets_from_local_directory(
                str(dataset_path),
                test_size=0.2,
                val_size=0.1,
                random_state=42,
                cache_dir=args.cache_dir
            )
            
            compare_datasets(train_dataset, val_dataset, test_dataset)
            
        except Exception as e:
            print(f"❌ Full dataset creation failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ Validation complete!")


if __name__ == "__main__":
    main()