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

try:
    from train.dataset import create_datasets_from_local_directory, create_data_loaders
    from train.data_converter import AttributionGraphConverter
except ImportError:
    from dataset import create_datasets_from_local_directory, create_data_loaders
    from data_converter import AttributionGraphConverter


def inspect_raw_graph(json_file_path):
    """Inspect a raw JSON attribution graph"""
    json_file_path = str(json_file_path)  # Convert Path to string
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


def deep_nan_detection(pyg_data):
    """Comprehensive NaN detection in PyG data"""
    print(f"\n🕵️ Deep NaN Detection Analysis:")
    
    nan_issues = 0
    
    # Check node features
    if hasattr(pyg_data, 'x') and pyg_data.x is not None:
        x = pyg_data.x
        node_nan_mask = torch.isnan(x)
        node_inf_mask = torch.isinf(x)
        
        total_nan_features = node_nan_mask.sum().item()
        total_inf_features = node_inf_mask.sum().item()
        
        print(f"  Node features ({x.shape}):")
        print(f"    NaN values: {total_nan_features}")
        print(f"    Inf values: {total_inf_features}")
        
        if total_nan_features > 0:
            nan_per_feature = node_nan_mask.sum(dim=0)
            for i, count in enumerate(nan_per_feature):
                if count > 0:
                    print(f"    Feature {i}: {count} NaN values")
            nan_issues += 1
        
        if total_inf_features > 0:
            inf_per_feature = node_inf_mask.sum(dim=0)
            for i, count in enumerate(inf_per_feature):
                if count > 0:
                    print(f"    Feature {i}: {count} Inf values")
            nan_issues += 1
        
        # Check for extreme values that might cause overflow
        extreme_large = (torch.abs(x) > 1000).sum().item()
        if extreme_large > 0:
            print(f"    ⚠️  {extreme_large} features with |value| > 1000")
            nan_issues += 1
    
    # Check edge attributes
    if hasattr(pyg_data, 'edge_attr') and pyg_data.edge_attr is not None:
        edge_attr = pyg_data.edge_attr
        edge_nan_mask = torch.isnan(edge_attr)
        edge_inf_mask = torch.isinf(edge_attr)
        
        total_nan_edges = edge_nan_mask.sum().item()
        total_inf_edges = edge_inf_mask.sum().item()
        
        print(f"  Edge attributes ({edge_attr.shape}):")
        print(f"    NaN values: {total_nan_edges}")
        print(f"    Inf values: {total_inf_edges}")
        
        if total_nan_edges > 0:
            print(f"    🚨 CRITICAL: Found NaN edge weights!")
            # Show which edges have NaN
            nan_edge_indices = edge_nan_mask.nonzero(as_tuple=False)
            print(f"    NaN edge locations: {nan_edge_indices[:5]}")  # Show first 5
            nan_issues += 1
        
        if total_inf_edges > 0:
            print(f"    🚨 CRITICAL: Found Inf edge weights!")
            inf_edge_indices = edge_inf_mask.nonzero(as_tuple=False)
            print(f"    Inf edge locations: {inf_edge_indices[:5]}")  # Show first 5
            nan_issues += 1
        
        # Check edge attribute statistics
        if edge_attr.numel() > 0:
            edge_flat = edge_attr.flatten()
            print(f"    Edge weight range: [{edge_flat.min():.6f}, {edge_flat.max():.6f}]")
            print(f"    Edge weight mean: {edge_flat.mean():.6f}")
            print(f"    Edge weight std: {edge_flat.std():.6f}")
            
            # Check for problematic gradients
            gradient_risk = edge_flat.std() / (abs(edge_flat.mean()) + 1e-8)
            if gradient_risk > 100:
                print(f"    ⚠️  High gradient explosion risk: std/mean = {gradient_risk:.2f}")
                nan_issues += 1
    
    # Check edge indices
    if hasattr(pyg_data, 'edge_index') and pyg_data.edge_index is not None:
        edge_index = pyg_data.edge_index
        print(f"  Edge indices ({edge_index.shape}):")
        
        if edge_index.numel() > 0:
            max_node_idx = edge_index.max().item()
            min_node_idx = edge_index.min().item()
            num_nodes = pyg_data.num_nodes if hasattr(pyg_data, 'num_nodes') else pyg_data.x.shape[0]
            
            print(f"    Node index range: [{min_node_idx}, {max_node_idx}]")
            print(f"    Expected node range: [0, {num_nodes-1}]")
            
            # Check for invalid node indices
            invalid_indices = (edge_index >= num_nodes) | (edge_index < 0)
            if invalid_indices.any():
                print(f"    🚨 CRITICAL: Found invalid node indices!")
                print(f"    Invalid count: {invalid_indices.sum().item()}")
                nan_issues += 1
    
    # Check labels
    if hasattr(pyg_data, 'y') and pyg_data.y is not None:
        y = pyg_data.y
        y_nan = torch.isnan(y).sum().item()
        y_inf = torch.isinf(y).sum().item()
        
        print(f"  Labels ({y.shape}):")
        print(f"    NaN values: {y_nan}")
        print(f"    Inf values: {y_inf}")
        print(f"    Label values: {y.unique().tolist()}")
        
        if y_nan > 0 or y_inf > 0:
            print(f"    🚨 CRITICAL: Invalid labels found!")
            nan_issues += 1
    
    if nan_issues == 0:
        print(f"  ✅ No NaN/Inf issues detected")
    else:
        print(f"  🚨 Found {nan_issues} critical issues that could cause NaN loss!")
    
    return nan_issues == 0


def verify_edge_attributes_detailed(original_edges, pyg_data, node_id_to_idx):
    """Comprehensive edge attribute validation with normalization analysis"""
    print(f"\n🔍 Detailed Edge Attribute Analysis:")
    
    edge_issues = 0
    
    # Collect all original edge weights for analysis
    original_weights = []
    converted_weights = pyg_data.edge_attr.flatten().tolist()
    
    for edge in original_edges:
        weight = edge.get('weight', edge.get('value', 1.0))
        if weight is not None:
            original_weights.append(float(weight))
    
    print(f"  Original edge weight statistics:")
    if original_weights:
        original_array = np.array(original_weights)
        print(f"    Count: {len(original_weights)}")
        print(f"    Range: [{original_array.min():.6f}, {original_array.max():.6f}]")
        print(f"    Mean: {original_array.mean():.6f}")
        print(f"    Median: {np.median(original_array):.6f}")
        print(f"    Std: {original_array.std():.6f}")
        
        # Percentiles for outlier analysis
        p99 = np.percentile(original_array, 99)
        p01 = np.percentile(original_array, 1)
        print(f"    1st/99th percentiles: [{p01:.6f}, {p99:.6f}]")
        
        # Check for problematic values
        zero_weights = sum(1 for w in original_weights if w == 0.0)
        negative_weights = sum(1 for w in original_weights if w < 0)
        large_weights = sum(1 for w in original_weights if abs(w) > 50)  # Based on analysis
        extreme_weights = sum(1 for w in original_weights if abs(w) > 100)
        tiny_weights = sum(1 for w in original_weights if 0 < abs(w) < 1e-6)
        
        print(f"    Zero weights: {zero_weights}")
        print(f"    Negative weights: {negative_weights} ({negative_weights/len(original_weights):.1%})")
        print(f"    Large weights (>50): {large_weights}")
        print(f"    Extreme weights (>100): {extreme_weights}")
        print(f"    Very tiny weights (<1e-6): {tiny_weights}")
        
        # Training stability assessment
        weight_range = original_array.max() - original_array.min()
        variance_ratio = original_array.std() / (abs(original_array.mean()) + 1e-8)
        
        print(f"    Weight range: {weight_range:.3f}")
        print(f"    Variance/mean ratio: {variance_ratio:.3f}")
        
        stability_issues = []
        if weight_range > 100:
            stability_issues.append("Large weight range")
        if variance_ratio > 10:
            stability_issues.append("High variance")
        if extreme_weights > 0:
            stability_issues.append("Extreme outliers")
        
        if stability_issues:
            print(f"    ⚠️  Training stability risks: {', '.join(stability_issues)}")
            edge_issues += 1
    
    print(f"  Converted (normalized) edge weight statistics:")
    if converted_weights:
        converted_array = np.array(converted_weights)
        print(f"    Count: {len(converted_weights)}")
        print(f"    Range: [{converted_array.min():.6f}, {converted_array.max():.6f}]")
        print(f"    Mean: {converted_array.mean():.6f}")
        print(f"    Median: {np.median(converted_array):.6f}")
        print(f"    Std: {converted_array.std():.6f}")
        
        # Check for numerical issues in converted weights
        nan_weights = sum(1 for w in converted_weights if np.isnan(w))
        inf_weights = sum(1 for w in converted_weights if np.isinf(w))
        zero_converted = sum(1 for w in converted_weights if abs(w) < 1e-8)
        negative_converted = sum(1 for w in converted_weights if w < 0)
        
        print(f"    NaN weights: {nan_weights}")
        print(f"    Inf weights: {inf_weights}")
        print(f"    Near-zero weights: {zero_converted}")
        print(f"    Negative weights preserved: {negative_converted} ({negative_converted/len(converted_weights):.1%})")
        
        if nan_weights > 0 or inf_weights > 0:
            print(f"    ❌ Found invalid converted edge weights!")
            edge_issues += 1
        else:
            print(f"    ✅ All converted weights are valid")
        
        # Check normalization effectiveness for training stability
        weight_range = converted_array.max() - converted_array.min()
        variance_ratio = converted_array.std() / (abs(converted_array.mean()) + 1e-8)
        
        print(f"    Normalized weight range: {weight_range:.3f}")
        print(f"    Normalized variance/mean ratio: {variance_ratio:.3f}")
        
        normalization_success = []
        if weight_range <= 20:  # Should be much smaller after normalization
            normalization_success.append("Good weight range")
        if variance_ratio <= 20:  # Should be reduced
            normalization_success.append("Controlled variance")
        if all(abs(w) <= 10 for w in converted_weights):  # No extreme outliers
            normalization_success.append("No extreme outliers")
        
        if len(normalization_success) >= 2:
            print(f"    ✅ Normalization effective: {', '.join(normalization_success)}")
        else:
            print(f"    ⚠️  Normalization may need adjustment")
            edge_issues += 1
        
        # Compare original vs normalized to show preservation of semantic information
        if original_weights and len(original_weights) == len(converted_weights):
            orig_neg_ratio = sum(1 for w in original_weights if w < 0) / len(original_weights)
            conv_neg_ratio = negative_converted / len(converted_weights)
            
            print(f"    Sign preservation: Original {orig_neg_ratio:.1%} negative → Converted {conv_neg_ratio:.1%} negative")
            
            if abs(orig_neg_ratio - conv_neg_ratio) < 0.01:  # Within 1%
                print(f"    ✅ Semantic information (sign) well preserved")
            else:
                print(f"    ⚠️  Some semantic information may be lost")
                edge_issues += 1
    
    # Check edge attribute tensor shape and dtype
    print(f"  Edge attribute tensor analysis:")
    print(f"    Shape: {pyg_data.edge_attr.shape}")
    print(f"    Dtype: {pyg_data.edge_attr.dtype}")
    print(f"    Requires grad: {pyg_data.edge_attr.requires_grad}")
    
    # Sample a few edge weights to verify conversion accuracy
    print(f"  Edge weight conversion verification:")
    verified_edges = 0
    for i, edge in enumerate(original_edges[:5]):  # Check first 5 edges
        source_id = edge.get('source')
        target_id = edge.get('target')
        
        if source_id in node_id_to_idx and target_id in node_id_to_idx:
            expected_src = node_id_to_idx[source_id]
            expected_dst = node_id_to_idx[target_id]
            expected_weight = float(edge.get('weight', edge.get('value', 1.0)))
            
            # Find corresponding edge in PyG data
            for j in range(pyg_data.edge_index.shape[1]):
                src_idx, dst_idx = pyg_data.edge_index[:, j]
                if src_idx == expected_src and dst_idx == expected_dst:
                    actual_weight = pyg_data.edge_attr[j].item()
                    weight_diff = abs(actual_weight - expected_weight)
                    
                    if weight_diff < 1e-6:
                        print(f"    ✅ Edge {i}: {source_id}->{target_id}, weight {actual_weight:.6f}")
                        verified_edges += 1
                    else:
                        print(f"    ❌ Edge {i}: {source_id}->{target_id}, expected {expected_weight:.6f}, got {actual_weight:.6f} (diff: {weight_diff:.6f})")
                        edge_issues += 1
                    break
    
    return edge_issues == 0


def verify_feature_conversion(json_file_path, converter, label):
    """Verify that features are correctly converted from JSON to PyG"""
    json_file_path = str(json_file_path)
    print(f"\n=== Feature Conversion Verification: {Path(json_file_path).name} ===")
    
    # Read and parse the original JSON
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Handle both formats
    if 'json' in data:
        json_string = data['json']
        original_graph = json.loads(json_string)
    elif 'nodes' in data and 'links' in data:
        json_string = json.dumps(data)
        original_graph = data
    else:
        print("❌ Unknown file format")
        return False
    
    # Convert to PyG
    pyg_data = converter.json_string_to_pyg_data(json_string, label=label)
    
    if pyg_data is None:
        print("❌ Conversion failed!")
        return False
    
    print(f"✅ Conversion successful: {pyg_data.num_nodes} nodes, {pyg_data.edge_index.shape[1]} edges")
    
    # Get original nodes (excluding error nodes)
    original_nodes = []
    for node in original_graph['nodes']:
        if 'node_id' in node and not converter._is_error_node(node):
            original_nodes.append(node)
    
    print(f"Original valid nodes: {len(original_nodes)}")
    print(f"PyG nodes: {pyg_data.num_nodes}")
    
    if len(original_nodes) != pyg_data.num_nodes:
        print("❌ Node count mismatch!")
        return False
    
    # Verify features for first few nodes
    print(f"\n🔍 Verifying node features for first 3 nodes:")
    feature_names = converter.feature_names
    
    mismatches = 0
    for i in range(min(3, len(original_nodes))):
        original_node = original_nodes[i]
        pyg_features = pyg_data.x[i]
        
        print(f"\nNode {i} ({original_node.get('node_id', 'unknown')}):")
        
        # Extract expected features manually
        expected_features = converter.extract_node_features(original_node)
        
        print(f"  Expected: {expected_features}")
        print(f"  PyG:      {pyg_features.tolist()}")
        
        # Compare each feature
        for j, (expected, actual, fname) in enumerate(zip(expected_features, pyg_features.tolist(), feature_names)):
            if abs(expected - actual) > 1e-6:
                print(f"    ❌ {fname}: expected {expected}, got {actual}")
                mismatches += 1
            else:
                print(f"    ✅ {fname}: {actual}")
    
    # Test edge preservation and validation
    print(f"\n🔍 Verifying edge conversion:")
    original_edges = original_graph.get('links', original_graph.get('edges', []))
    
    # Create node ID to index mapping
    node_id_to_idx = {}
    for i, node in enumerate(original_nodes):
        node_id_to_idx[node['node_id']] = i
    
    # Count valid original edges
    valid_original_edges = 0
    for edge in original_edges:
        source_id = edge.get('source')
        target_id = edge.get('target')
        if source_id in node_id_to_idx and target_id in node_id_to_idx:
            valid_original_edges += 1
    
    print(f"  Original valid edges: {valid_original_edges}")
    print(f"  PyG edges: {pyg_data.edge_index.shape[1]}")
    
    if valid_original_edges != pyg_data.edge_index.shape[1]:
        print(f"    ❌ Edge count mismatch!")
        mismatches += 1
    else:
        print(f"    ✅ Edge count matches")
    
    # CRITICAL: Deep NaN detection first
    nan_detection_passed = deep_nan_detection(pyg_data)
    if not nan_detection_passed:
        mismatches += 1

    # Enhanced edge validation
    if pyg_data.edge_index.shape[1] > 0:
        edge_validation_passed = verify_edge_attributes_detailed(original_edges, pyg_data, node_id_to_idx)
        if not edge_validation_passed:
            mismatches += 1
    else:
        print(f"    ❌ Graph has no edges! This should not happen for attribution graphs.")
        mismatches += 1
    
    if mismatches == 0:
        print(f"\n✅ All feature and edge verifications passed!")
        return True
    else:
        print(f"\n❌ Found {mismatches} validation issues!")
        return False


def inspect_pyg_conversion(json_file_path, converter, label):
    """Test conversion of a single file to PyG"""
    json_file_path = str(json_file_path)  # Convert Path to string
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
    
    # Test PyG conversion with detailed feature verification
    print(f"\n3️⃣ Testing PyG conversion with feature verification...")
    
    verification_passed = 0
    total_tests = 0
    
    for i, file_path in enumerate(benign_files[:2]):  # Test 2 benign files
        total_tests += 1
        if verify_feature_conversion(file_path, converter, label=0):
            verification_passed += 1
    
    for i, file_path in enumerate(injected_files[:2]):  # Test 2 injected files
        total_tests += 1
        if verify_feature_conversion(file_path, converter, label=1):
            verification_passed += 1
    
    print(f"\n🎯 Feature Verification Summary: {verification_passed}/{total_tests} files passed all checks")
    
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