#!/usr/bin/env python3
"""
Script to debug model collapse - investigate why model always predicts class 1
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import Counter

try:
    from train.dataset import create_datasets_from_local_directory, create_data_loaders
    from train.models import PromptInjectionGraphGPS
except ImportError:
    from dataset import create_datasets_from_local_directory, create_data_loaders
    from models import PromptInjectionGraphGPS


def analyze_dataset_separability(train_dataset, val_dataset):
    """Analyze if the two classes are actually separable"""
    print("\n=== Dataset Separability Analysis ===")
    
    # Collect features from both classes
    benign_features = []
    injected_features = []
    benign_graph_stats = []
    injected_graph_stats = []
    
    for data in list(train_dataset) + list(val_dataset):
        label = data.y.item()
        features = data.x  # [num_nodes, num_features]
        
        # Graph-level statistics
        graph_stats = {
            'num_nodes': data.num_nodes,
            'num_edges': data.edge_index.shape[1],
            'avg_degree': data.edge_index.shape[1] * 2 / data.num_nodes if data.num_nodes > 0 else 0,
            'feature_means': features.mean(dim=0).numpy(),
            'feature_stds': features.std(dim=0).numpy(),
            'feature_mins': features.min(dim=0)[0].numpy(),
            'feature_maxs': features.max(dim=0)[0].numpy(),
        }
        
        if label == 0:  # Benign
            benign_features.append(features.mean(dim=0))  # Graph-level feature average
            benign_graph_stats.append(graph_stats)
        else:  # Injected
            injected_features.append(features.mean(dim=0))
            injected_graph_stats.append(graph_stats)
    
    # Convert to tensors
    benign_features = torch.stack(benign_features) if benign_features else torch.empty(0, 9)
    injected_features = torch.stack(injected_features) if injected_features else torch.empty(0, 9)
    
    print(f"Benign graphs: {len(benign_graph_stats)}")
    print(f"Injected graphs: {len(injected_graph_stats)}")
    
    if len(benign_graph_stats) == 0 or len(injected_graph_stats) == 0:
        print("❌ Missing one class - dataset is imbalanced!")
        return False
    
    # Compare graph structure
    print(f"\nGraph Structure Comparison:")
    benign_nodes = [s['num_nodes'] for s in benign_graph_stats]
    injected_nodes = [s['num_nodes'] for s in injected_graph_stats]
    benign_edges = [s['num_edges'] for s in benign_graph_stats]
    injected_edges = [s['num_edges'] for s in injected_graph_stats]
    
    print(f"  Nodes - Benign: {np.mean(benign_nodes):.1f}±{np.std(benign_nodes):.1f}, Injected: {np.mean(injected_nodes):.1f}±{np.std(injected_nodes):.1f}")
    print(f"  Edges - Benign: {np.mean(benign_edges):.1f}±{np.std(benign_edges):.1f}, Injected: {np.mean(injected_edges):.1f}±{np.std(injected_edges):.1f}")
    
    # Feature-level comparison
    print(f"\nFeature-level Comparison (graph averages):")
    feature_names = ['influence', 'activation', 'layer', 'ctx_idx', 'feature', 'is_cross_layer_transcoder', 'is_mlp_error', 'is_embedding', 'is_target_logit']
    
    separable_features = 0
    for i, fname in enumerate(feature_names):
        benign_vals = benign_features[:, i]
        injected_vals = injected_features[:, i]
        
        benign_mean = benign_vals.mean().item()
        injected_mean = injected_vals.mean().item()
        benign_std = benign_vals.std().item()
        injected_std = injected_vals.std().item()
        
        # Simple separability test - are means far apart compared to stds?
        separation = abs(benign_mean - injected_mean) / (benign_std + injected_std + 1e-8)
        
        print(f"  {fname:25}: Benign={benign_mean:8.4f}±{benign_std:.4f}, Injected={injected_mean:8.4f}±{injected_std:.4f}, Sep={separation:.3f}")
        
        if separation > 0.5:  # Somewhat separable
            separable_features += 1
    
    print(f"\nSeparable features: {separable_features}/{len(feature_names)}")
    
    if separable_features == 0:
        print("❌ No features show clear separation between classes!")
        return False
    else:
        print(f"✅ {separable_features} features show some separation")
        return True


def test_simple_model(train_loader, val_loader, device):
    """Test with a much simpler model to see if it can learn"""
    print("\n=== Testing Simple Model ===")
    
    # Get feature dimension from first batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch.x.shape[1]
    
    # Very simple model: just MLP on graph-level averages
    class SimpleClassifier(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 2)
            )
        
        def forward(self, x, edge_index, batch, edge_attr=None):
            # Global mean pooling
            from torch_geometric.nn import global_mean_pool
            x_global = global_mean_pool(x, batch)
            return self.fc(x_global), None
    
    model = SimpleClassifier(input_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Simple model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train for a few epochs
    model.train()
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, _ = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
                pred = logits.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)
        
        train_acc = correct / total
        val_acc = val_correct / val_total
        
        print(f"  Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        model.train()
    
    if val_acc > 0.6:
        print("✅ Simple model can learn - data is separable!")
        return True
    else:
        print("❌ Even simple model struggles - data may not be separable")
        return False


def analyze_training_dynamics(model, train_loader, device):
    """Analyze what happens during the first few training steps"""
    print("\n=== Training Dynamics Analysis ===")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Get a few batches
    batches = []
    for i, batch in enumerate(train_loader):
        batches.append(batch.to(device))
        if i >= 3:  # Just analyze first few batches
            break
    
    print("Analyzing first training steps...")
    
    model.train()
    for step in range(5):
        for batch_idx, batch in enumerate(batches):
            
            # Forward pass
            logits, _ = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            loss = criterion(logits, batch.y)
            
            # Analyze predictions
            probs = torch.softmax(logits, dim=1)
            pred_classes = logits.argmax(dim=1)
            
            true_labels = batch.y
            class_0_count = (true_labels == 0).sum().item()
            class_1_count = (true_labels == 1).sum().item()
            pred_0_count = (pred_classes == 0).sum().item()
            pred_1_count = (pred_classes == 1).sum().item()
            
            avg_prob_0 = probs[:, 0].mean().item()
            avg_prob_1 = probs[:, 1].mean().item()
            
            print(f"  Step {step+1}, Batch {batch_idx+1}:")
            print(f"    True: {class_0_count} class 0, {class_1_count} class 1")
            print(f"    Pred: {pred_0_count} class 0, {pred_1_count} class 1")
            print(f"    Avg probs: {avg_prob_0:.3f} class 0, {avg_prob_1:.3f} class 1")
            print(f"    Loss: {loss.item():.4f}")
            
            # Check for gradient issues
            optimizer.zero_grad()
            loss.backward()
            
            total_grad_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            print(f"    Gradient norm: {total_grad_norm:.6f}")
            
            optimizer.step()
            
            if batch_idx >= 1:  # Only analyze first 2 batches per step
                break
        print()


def check_data_loading(train_loader, val_loader):
    """Check if data loading is correct"""
    print("\n=== Data Loading Check ===")
    
    # Check class distribution
    train_labels = []
    val_labels = []
    
    for batch in train_loader:
        train_labels.extend(batch.y.tolist())
    
    for batch in val_loader:
        val_labels.extend(batch.y.tolist())
    
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)
    
    print(f"Train set: {dict(train_counts)}")
    print(f"Val set: {dict(val_counts)}")
    
    # Check for data leakage or weird patterns
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    
    train_ratio = train_counts[1] / total_train if total_train > 0 else 0
    val_ratio = val_counts[1] / total_val if total_val > 0 else 0
    
    print(f"Class 1 ratio - Train: {train_ratio:.3f}, Val: {val_ratio:.3f}")
    
    if abs(train_ratio - val_ratio) > 0.1:
        print("⚠️  Large class ratio difference between train/val")
    
    if train_ratio == 0 or train_ratio == 1:
        print("❌ Train set has only one class!")
        return False
    
    if val_ratio == 0 or val_ratio == 1:
        print("❌ Val set has only one class!")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Debug model collapse')
    parser.add_argument('dataset_dir', help='Directory containing benign/ and injected/ subdirectories')
    parser.add_argument('--cache_dir', default='./debug_cache', help='Cache directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("🐛 Model Collapse Debugging")
    print("=" * 50)
    
    # Load datasets
    print("1️⃣ Loading datasets...")
    train_dataset, val_dataset, test_dataset, converter = create_datasets_from_local_directory(
        args.dataset_dir,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        cache_dir=args.cache_dir
    )
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    print(f"Datasets loaded: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Run diagnostics
    check_data_loading(train_loader, val_loader)
    
    separable = analyze_dataset_separability(train_dataset, val_dataset)
    
    if separable:
        simple_works = test_simple_model(train_loader, val_loader, device)
        
        if simple_works:
            # Test the actual model
            print("\n=== Testing GraphGPS Model ===")
            model = PromptInjectionGraphGPS(
                input_dim=converter.get_feature_dim(),
                hidden_dim=64,  # Smaller for debugging
                num_layers=3,
                num_heads=4,
                num_classes=2,
                dropout=0.1
            ).to(device)
            
            analyze_training_dynamics(model, train_loader, device)
    
    print("\n🎯 Debugging Summary:")
    print("- Check the separability analysis for feature differences")
    print("- If features aren't separable, the data might be the issue")
    print("- If simple model works but GraphGPS doesn't, it's a model architecture issue")
    print("- Look at training dynamics for gradient/learning rate issues")


if __name__ == "__main__":
    main()