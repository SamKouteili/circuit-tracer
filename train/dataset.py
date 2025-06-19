"""
Clean dataset classes and data loading utilities for prompt injection detection
"""

import torch
from torch_geometric.data import Dataset, DataLoader, Data
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional
import os
from train.data_converter import AttributionGraphConverter


class PromptInjectionDataset(Dataset):
    """PyTorch Geometric dataset for prompt injection detection"""

    def __init__(self, data_list: List[Data]):
        super().__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def __repr__(self):
        return f"PromptInjectionDataset({len(self)} graphs)"


def create_datasets_from_huggingface(
    dataset_name: str = "samkouteili/injection-attribution-graphs",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[PromptInjectionDataset, PromptInjectionDataset, PromptInjectionDataset, AttributionGraphConverter]:
    """
    Create train/val/test datasets from Hugging Face dataset using local conversion
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        test_size: Fraction of data to use for testing
        val_size: Fraction of remaining data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, converter)
    """
    
    print(f"Loading dataset '{dataset_name}' using local conversion...")
    
    from convert_and_load_dataset import download_and_convert_dataset
    
    # Download and convert dataset
    benign_files, injected_files = download_and_convert_dataset(dataset_name)
    
    if not benign_files or not injected_files:
        raise ValueError("No files were successfully downloaded and converted")
    
    print(f"Using converted files: {len(benign_files)} benign, {len(injected_files)} injected")
    
    # Use the converted files with proper JSON string handling
    return create_datasets_from_converted_files(
        benign_files=benign_files,
        injected_files=injected_files,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )


def create_datasets_from_converted_files(benign_files, injected_files, test_size=0.2, val_size=0.1, random_state=42):
    """Create datasets from converted files that have JSON strings"""
    
    print(f"Creating datasets from {len(benign_files)} benign and {len(injected_files)} injected converted files...")
    
    # Initialize converter
    converter = AttributionGraphConverter()
    
    # Collect all JSON strings from converted files
    all_json_strings = []
    
    for file_path in benign_files + injected_files:
        try:
            import json as json_module
            with open(file_path, 'r') as f:
                converted_data = json_module.load(f)
            if 'json' in converted_data:
                all_json_strings.append(converted_data['json'])
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    # Build vocabulary from JSON strings
    vocab_success = converter.build_vocabulary_from_json_strings(all_json_strings)
    
    if not vocab_success:
        raise ValueError("Failed to build vocabulary from converted files")
    
    # Convert all data to PyG Data objects
    all_data = []
    conversion_stats = {'benign': {'success': 0, 'failed': 0}, 
                       'injected': {'success': 0, 'failed': 0}}
    
    # Process benign files (label=0)
    for file_path in benign_files:
        try:
            import json as json_module
            with open(file_path, 'r') as f:
                converted_data = json_module.load(f)
            if 'json' in converted_data:
                data = converter.json_string_to_pyg_data(converted_data['json'], label=0)
                if data is not None:
                    all_data.append(data)
                    conversion_stats['benign']['success'] += 1
                else:
                    conversion_stats['benign']['failed'] += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            conversion_stats['benign']['failed'] += 1
    
    # Process injected files (label=1)
    for file_path in injected_files:
        try:
            import json as json_module
            with open(file_path, 'r') as f:
                converted_data = json_module.load(f)
            if 'json' in converted_data:
                data = converter.json_string_to_pyg_data(converted_data['json'], label=1)
                if data is not None:
                    all_data.append(data)
                    conversion_stats['injected']['success'] += 1
                else:
                    conversion_stats['injected']['failed'] += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            conversion_stats['injected']['failed'] += 1
    
    print(f"Conversion results:")
    print(f"  Benign graphs: {conversion_stats['benign']['success']} success, {conversion_stats['benign']['failed']} failed")
    print(f"  Injected graphs: {conversion_stats['injected']['success']} success, {conversion_stats['injected']['failed']} failed")
    
    if len(all_data) == 0:
        raise ValueError("No graphs were successfully converted!")
    
    # Create labels for stratification
    labels = [data.y.item() for data in all_data]
    
    # Check class balance
    num_benign = sum(1 for label in labels if label == 0)
    num_injected = sum(1 for label in labels if label == 1)
    
    print(f"Dataset composition: {num_benign} benign, {num_injected} injected graphs")
    
    if num_benign == 0 or num_injected == 0:
        raise ValueError("Dataset must contain both benign and injected graphs!")
    
    # Split data
    if len(all_data) < 3:
        print("Warning: Very small dataset, using simple split")
        train_data = all_data[:-1] if len(all_data) > 1 else all_data
        val_data = []
        test_data = all_data[-1:] if len(all_data) > 1 else []
    else:
        # Stratified splits to maintain class balance
        train_val_data, test_data, train_val_labels, test_labels = train_test_split(
            all_data, labels, 
            test_size=test_size, 
            stratify=labels, 
            random_state=random_state
        )
        
        if len(train_val_data) > 2 and val_size > 0:
            train_data, val_data, _, _ = train_test_split(
                train_val_data, train_val_labels,
                test_size=val_size / (1 - test_size),
                stratify=train_val_labels,
                random_state=random_state
            )
        else:
            train_data = train_val_data
            val_data = []
    
    # Create datasets
    train_dataset = PromptInjectionDataset(train_data)
    val_dataset = PromptInjectionDataset(val_data) if val_data else PromptInjectionDataset([])
    test_dataset = PromptInjectionDataset(test_data) if test_data else PromptInjectionDataset([])
    
    print(f"Dataset splits:")
    print(f"  Train: {len(train_dataset)} graphs")
    print(f"  Val: {len(val_dataset)} graphs") 
    print(f"  Test: {len(test_dataset)} graphs")
    
    return train_dataset, val_dataset, test_dataset, converter


def create_data_loaders(
    train_dataset: PromptInjectionDataset,
    val_dataset: PromptInjectionDataset,
    test_dataset: PromptInjectionDataset,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training"""

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    ) if len(val_dataset) > 0 else None

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    ) if len(test_dataset) > 0 else None

    return train_loader, val_loader, test_loader


def test_dataset_creation():
    """Test dataset creation with local conversion"""
    print("Testing dataset creation...")

    try:
        # Download and convert the small dataset
        train_dataset, val_dataset, test_dataset, converter = create_datasets_from_huggingface(
            dataset_name="samkouteili/injection-attribution-graphs-small",
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )
        
        print(f"✅ Successfully created datasets!")
        print(f"  Feature dimension: {converter.get_feature_dim()}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset, batch_size=2
        )
        
        print(f"✅ Successfully created data loaders!")
        
        # Test loading a batch
        if len(train_loader) > 0:
            batch = next(iter(train_loader))
            print(f"Sample batch:")
            print(f"  Batch size: {batch.num_graphs}")
            print(f"  Node features shape: {batch.x.shape}")
            print(f"  Edge index shape: {batch.edge_index.shape}")
            print(f"  Labels: {batch.y}")
            print(f"  Batch tensor: {batch.batch.shape}")

        return True
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_dataset_creation()