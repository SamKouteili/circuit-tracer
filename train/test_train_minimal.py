#!/usr/bin/env python3
"""
Minimal training test to verify everything works
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test training pipeline')
    parser.add_argument('--dataset', type=str, default='samkouteili/injection-attribution-graphs-small')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    
    args = parser.parse_args()
    
    print(f"🚀 Testing training pipeline...")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Test imports
    try:
        from dataset import create_datasets_from_huggingface, create_data_loaders
        from models import PromptInjectionGraphGPS
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test dataset loading
    try:
        print("\n1️⃣ Testing dataset loading...")
        train_dataset, val_dataset, test_dataset, converter = create_datasets_from_huggingface(
            dataset_name=args.dataset,
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )
        print(f"✅ Dataset loaded: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        print(f"✅ Feature dimension: {converter.get_feature_dim()}")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test data loaders
    try:
        print("\n2️⃣ Testing data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=args.batch_size,
            num_workers=0
        )
        print(f"✅ Data loaders created")
    except Exception as e:
        print(f"❌ Data loader creation failed: {e}")
        return False
    
    # Test model creation
    try:
        print("\n3️⃣ Testing model creation...")
        model = PromptInjectionGraphGPS(
            input_dim=converter.get_feature_dim(),
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            num_classes=2,
            dropout=0.1
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created with {total_params:,} parameters")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test forward pass
    try:
        print("\n4️⃣ Testing forward pass...")
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            with torch.no_grad():
                logits, embeddings = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            print(f"✅ Forward pass successful: {logits.shape}")
        else:
            print("⚠️ No training batches available")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All tests passed! Full training should work.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)