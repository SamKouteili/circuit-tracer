"""
Training script for prompt injection detection using GraphGPS on attribution graphs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from train.dataset import create_datasets_from_huggingface, create_data_loaders
from train.models import PromptInjectionGraphGPS

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

def compute_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute classification metrics"""
    pred_classes = torch.argmax(predictions, dim=1)
    correct = (pred_classes == labels).float()
    accuracy = correct.mean().item()
    
    # Compute per-class metrics
    num_classes = predictions.size(1)
    metrics = {'accuracy': accuracy}
    
    for class_idx in range(num_classes):
        class_mask = (labels == class_idx)
        if class_mask.sum() > 0:
            class_correct = correct[class_mask].sum().item()
            class_total = class_mask.sum().item()
            metrics[f'class_{class_idx}_accuracy'] = class_correct / class_total
        else:
            metrics[f'class_{class_idx}_accuracy'] = 0.0
    
    return metrics

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, print_freq=10):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, _ = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
        loss = criterion(logits, batch.y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        all_predictions.append(logits.detach().cpu())
        all_labels.append(batch.y.detach().cpu())
        
        if batch_idx % print_freq == 0:
            print(f"    Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    # Compute epoch metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_predictions, all_labels)
    metrics['loss'] = running_loss / len(train_loader)
    
    return metrics

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            
            logits, _ = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            loss = criterion(logits, batch.y)
            
            running_loss += loss.item()
            all_predictions.append(logits.detach().cpu())
            all_labels.append(batch.y.detach().cpu())
    
    # Compute validation metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_predictions, all_labels)
    metrics['loss'] = running_loss / len(val_loader)
    
    return metrics

def test_model(model, test_loader, criterion, device):
    """Test the model"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            logits, _ = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            loss = criterion(logits, batch.y)
            
            probabilities = torch.softmax(logits, dim=1)
            
            running_loss += loss.item()
            all_predictions.append(logits.detach().cpu())
            all_labels.append(batch.y.detach().cpu())
            all_probabilities.append(probabilities.detach().cpu())
    
    # Compute test metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_probabilities = torch.cat(all_probabilities, dim=0)
    
    metrics = compute_metrics(all_predictions, all_labels)
    metrics['loss'] = running_loss / len(test_loader)
    
    return metrics, all_probabilities, all_labels

def save_checkpoint(model, optimizer, scheduler, epoch, train_metrics, val_metrics, checkpoint_path):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"    Checkpoint saved: {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description='Train GraphGPS for prompt injection detection')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='samkouteili/injection-attribution-graphs',
                       help='HuggingFace dataset name')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set fraction')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Validation set fraction')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of GraphGPS layers')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--output_dir', type=str, default='./training_output',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 Starting training on {device}")
    print(f"Arguments: {vars(args)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training configuration
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # 1. Load datasets
    print(f"\n1️⃣ Loading datasets...")
    start_time = time.time()
    
    train_dataset, val_dataset, test_dataset, converter = create_datasets_from_huggingface(
        dataset_name=args.dataset,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )
    
    load_time = time.time() - start_time
    print(f"✅ Datasets loaded in {load_time:.2f}s")
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"   Feature dimension: {converter.get_feature_dim()}")
    
    # 2. Create data loaders
    print(f"\n2️⃣ Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"✅ Data loaders created")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader) if val_loader else 0}")
    print(f"   Test batches: {len(test_loader) if test_loader else 0}")
    
    # 3. Initialize model
    print(f"\n3️⃣ Initializing model...")
    model = PromptInjectionGraphGPS(
        input_dim=converter.get_feature_dim(),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_classes=2,
        dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model initialized")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB")
    
    # 4. Setup training components
    print(f"\n4️⃣ Setting up training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=args.patience)
    
    print(f"✅ Training setup complete")
    print(f"   Loss: CrossEntropyLoss")
    print(f"   Optimizer: Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"   Scheduler: ReduceLROnPlateau")
    print(f"   Early stopping: {args.patience} epochs patience")
    
    # 5. Training loop
    print(f"\n5️⃣ Starting training loop...")
    training_history = {'train': [], 'val': []}
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        if val_loader:
            val_metrics = validate_epoch(model, val_loader, criterion, device)
        else:
            val_metrics = {'loss': float('inf'), 'accuracy': 0.0}
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"  Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save training history
        training_history['train'].append(train_metrics)
        training_history['val'].append(val_metrics)
        
        # Save checkpoint for best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_checkpoint_path = output_dir / 'best_model.pt'
            save_checkpoint(model, optimizer, scheduler, epoch, train_metrics, val_metrics, best_checkpoint_path)
        
        # Save regular checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
            save_checkpoint(model, optimizer, scheduler, epoch, train_metrics, val_metrics, checkpoint_path)
        
        # Early stopping check
        if early_stopping(val_metrics['loss'], model):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # 6. Test evaluation
    print(f"\n6️⃣ Final evaluation...")
    
    # Load best model for testing
    best_checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    if test_loader:
        test_metrics, test_probabilities, test_labels = test_model(model, test_loader, criterion, device)
        print(f"✅ Test evaluation complete")
        print(f"   Test Loss: {test_metrics['loss']:.4f}")
        print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
        for key, value in test_metrics.items():
            if key.startswith('class_'):
                print(f"   {key}: {value:.4f}")
    else:
        test_metrics = {}
        print("⚠️  No test data available")
    
    # 7. Save final results
    print(f"\n7️⃣ Saving results...")
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        # Convert tensors to lists for JSON serialization
        history_serializable = {}
        for split, epochs in training_history.items():
            history_serializable[split] = []
            for epoch_metrics in epochs:
                epoch_dict = {}
                for key, value in epoch_metrics.items():
                    if isinstance(value, torch.Tensor):
                        epoch_dict[key] = value.item()
                    else:
                        epoch_dict[key] = value
                history_serializable[split].append(epoch_dict)
        json.dump(history_serializable, f, indent=2)
    
    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': converter.get_feature_dim(),
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'num_classes': 2,
            'dropout': args.dropout
        },
        'test_metrics': test_metrics,
        'training_args': vars(args)
    }, final_model_path)
    
    print(f"✅ Training complete!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Output directory: {output_dir}")
    print(f"   Best model: {output_dir / 'best_model.pt'}")
    print(f"   Final model: {final_model_path}")
    print(f"   Training history: {history_path}")

if __name__ == "__main__":
    main()