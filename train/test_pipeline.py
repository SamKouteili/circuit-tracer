"""
Test script to validate the entire data pipeline
"""

import sys
import os
import torch

# Add the parent directory to path so we can import from train module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_full_pipeline():
    """Test the complete data processing pipeline"""
    print("=" * 60)
    print("TESTING PROMPT INJECTION DETECTION PIPELINE")
    print("=" * 60)
    
    # Test 1: Data Converter
    print("\n1. Testing Data Converter...")
    try:
        from train.data_converter import test_converter
        converter_success = test_converter()
    except Exception as e:
        print(f"❌ Data converter test failed: {e}")
        return False
    
    if not converter_success:
        print("❌ Data converter test failed!")
        return False
    
    # Test 2: Dataset Creation
    print("\n2. Testing Dataset Creation...")
    try:
        from train.dataset import test_dataset_creation
        dataset_success = test_dataset_creation()
    except Exception as e:
        print(f"❌ Dataset creation test failed: {e}")
        return False
    
    if not dataset_success:
        print("❌ Dataset creation test failed!")
        return False
    
    # Test 3: Model
    print("\n3. Testing Model...")
    try:
        from train.models import test_model
        model_success = test_model()
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    if not model_success:
        print("❌ Model test failed!")
        return False
    
    # Test 4: Integration Test (End-to-End)
    print("\n4. Testing End-to-End Integration...")
    try:
        integration_success = test_integration()
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    if not integration_success:
        print("❌ Integration test failed!")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED! Pipeline is ready for training.")
    print("=" * 60)
    return True

def test_integration():
    """Test that all components work together"""
    print("Running integration test...")
    
    try:
        from train.data_converter import AttributionGraphConverter
        from train.dataset import create_datasets_from_files, create_data_loaders
        from train.models import PromptInjectionGraphGPS
        
        # Get paths to JSON files
        base_path = "/Users/samkouteili/rose/circuits"
        normal_files = [os.path.join(base_path, "Ellen Pompeo Attribution Graph.json")]
        injection_files = [
            os.path.join(base_path, "Ellen Pompeo Prompt Injection.json"),
            os.path.join(base_path, "Prompt Injection Cyfyre.json")
        ]
        
        # Create datasets
        train_dataset, val_dataset, test_dataset, converter = create_datasets_from_files(
            normal_files=normal_files,
            injection_files=injection_files,
            test_size=0.3,
            val_size=0.0,
            random_state=42
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset, batch_size=2
        )
        
        # Create model
        model = PromptInjectionGraphGPS(
            input_dim=converter.get_feature_dim(),
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            num_classes=2,
            dropout=0.1
        )
        
        # Test training step
        model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Get a batch
        batch = next(iter(train_loader))
        
        # Forward pass
        logits, node_embeddings = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        
        # Compute loss
        loss = criterion(logits, batch.y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✅ Integration test successful!")
        print(f"  Batch processed: {batch.num_graphs} graphs")
        print(f"  Model output shape: {logits.shape}")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Gradients computed successfully")
        
        # Test inference
        model.eval()
        with torch.no_grad():
            logits_eval, _ = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            predictions = torch.argmax(logits_eval, dim=1)
            
        print(f"  Inference test successful!")
        print(f"  Predictions: {predictions.tolist()}")
        print(f"  True labels: {batch.y.tolist()}")
        
        return True
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)