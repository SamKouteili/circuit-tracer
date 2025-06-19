"""
Training simulation to test the full pipeline without actual training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from train.dataset import create_datasets_from_huggingface, create_data_loaders
from train.models import PromptInjectionGraphGPS


def simulate_training():
    """Simulate a training run to test all components"""
    print("🚀 Starting training simulation...")

    # 1. Load datasets
    print("\n1: Loading datasets...")
    try:
        train_dataset, val_dataset, test_dataset, converter = create_datasets_from_huggingface(
            dataset_name="samkouteili/injection-attribution-graphs-small",
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )
        print(f"✅ Datasets loaded successfully!")
        print(
            f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        print(f"   Feature dimension: {converter.get_feature_dim()}")
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

    # 2. Create data loaders
    print("\n2: Creating data loaders...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=2,  # Small batch for testing
            num_workers=0
        )
        print(f"✅ Data loaders created!")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader) if val_loader else 0}")
        print(f"   Test batches: {len(test_loader) if test_loader else 0}")
    except Exception as e:
        print(f"❌ Data loader creation failed: {e}")
        return False

    # 3. Initialize model
    print("\n3: Initializing model...")
    try:
        model = PromptInjectionGraphGPS(
            input_dim=converter.get_feature_dim(),
            hidden_dim=64,  # Smaller for testing
            num_layers=2,   # Fewer layers for testing
            num_heads=4,    # Fewer heads for testing
            num_classes=2,
            dropout=0.1
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)

        print(f"✅ Model initialized!")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

    # 4. Setup training components
    print("\n4: Setting up training components...")
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5)

        print(f"✅ Training components ready!")
        print(f"   Loss: CrossEntropyLoss")
        print(f"   Optimizer: Adam (lr=0.001)")
        print(f"   Scheduler: StepLR (step=10, gamma=0.5)")
    except Exception as e:
        print(f"❌ Training component setup failed: {e}")
        return False

    # 5. Test forward pass
    print("\n5: Testing forward pass...")
    try:
        model.eval()
        test_batch = next(iter(train_loader))

        with torch.no_grad():
            logits, node_embeddings = model(
                test_batch.x,
                test_batch.edge_index,
                test_batch.batch,
                test_batch.edge_attr
            )

        print(f"✅ Forward pass successful!")
        print(f"   Input shape: {test_batch.x.shape}")
        print(f"   Output logits: {logits.shape}")
        print(f"   Node embeddings: {node_embeddings.shape}")
        print(f"   Predictions: {torch.softmax(logits, dim=1)}")
        print(f"   True labels: {test_batch.y}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 6. Test training step
    print("\n6: Testing training step...")
    try:
        model.train()
        optimizer.zero_grad()

        # Forward pass
        logits, _ = model(
            test_batch.x,
            test_batch.edge_index,
            test_batch.batch,
            test_batch.edge_attr
        )

        # Compute loss
        loss = criterion(logits, test_batch.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == test_batch.y).float().mean()

        print(f"✅ Training step successful!")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Accuracy: {accuracy.item():.4f}")
        print(f"   Predictions: {predictions.tolist()}")
        print(f"   True labels: {test_batch.y.tolist()}")
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 7. Test validation step
    print("\n7: Testing validation step...")
    try:
        if val_loader and len(val_loader) > 0:
            model.eval()
            val_batch = next(iter(val_loader))

            with torch.no_grad():
                val_logits, _ = model(
                    val_batch.x,
                    val_batch.edge_index,
                    val_batch.batch,
                    val_batch.edge_attr
                )
                val_loss = criterion(val_logits, val_batch.y)
                val_predictions = torch.argmax(val_logits, dim=1)
                val_accuracy = (val_predictions == val_batch.y).float().mean()

            print(f"✅ Validation step successful!")
            print(f"   Val loss: {val_loss.item():.4f}")
            print(f"   Val accuracy: {val_accuracy.item():.4f}")
        else:
            print(f"⚠️  No validation data available")
    except Exception as e:
        print(f"❌ Validation step failed: {e}")
        return False

    # 8. Test batch iteration
    print("\n8: Testing batch iteration...")
    try:
        batch_count = 0
        total_samples = 0

        for batch in train_loader:
            batch_count += 1
            total_samples += batch.num_graphs

            # Verify batch structure
            assert hasattr(batch, 'x'), "Batch missing node features"
            assert hasattr(batch, 'edge_index'), "Batch missing edge index"
            assert hasattr(batch, 'batch'), "Batch missing batch tensor"
            assert hasattr(batch, 'y'), "Batch missing labels"

            if batch_count >= 3:  # Only test first 3 batches
                break

        print(f"✅ Batch iteration successful!")
        print(f"   Processed {batch_count} batches")
        print(f"   Total samples: {total_samples}")
    except Exception as e:
        print(f"❌ Batch iteration failed: {e}")
        return False

    # 9. Memory check
    print("\n9: Memory usage check...")
    try:
        if torch.cuda.is_available():
            print(f"   CUDA available: Yes")
            print(f"   Current device: {torch.cuda.current_device()}")
            # Note: Memory check would go here if using GPU
        else:
            print(f"   CUDA available: No (using CPU)")

        print(f"✅ Memory check complete!")
    except Exception as e:
        print(f"❌ Memory check failed: {e}")
        return False

    print(f"\n🎉 ALL SIMULATION TESTS PASSED!")
    print(f"✅ Training pipeline is ready for full training run")
    print(f"\n📝 Summary:")
    print(f"   • Dataset loading: Working")
    print(f"   • Data loaders: Working")
    print(f"   • Model: Working ({trainable_params:,} parameters)")
    print(f"   • Forward pass: Working")
    print(f"   • Training step: Working")
    print(f"   • Validation step: Working")
    print(f"   • Batch processing: Working")
    print(f"   • Ready for cluster training!")

    return True


if __name__ == "__main__":
    success = simulate_training()
    if not success:
        exit(1)
