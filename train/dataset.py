"""
Clean dataset classes and data loading utilities for prompt injection detection
"""

import torch
from torch_geometric.data import Dataset, DataLoader, Data
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional
import os
from tqdm import tqdm
import json


try:
    from train.data_converter import AttributionGraphConverter
except ImportError:
    from data_converter import AttributionGraphConverter


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

    print(
        f"Using converted files: {len(benign_files)} benign, {len(injected_files)} injected")

    # Use the converted files with proper JSON string handling
    return create_datasets_from_converted_files(
        benign_files=benign_files,
        injected_files=injected_files,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )


def create_datasets_from_local_directory(
    dataset_path: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[PromptInjectionDataset, PromptInjectionDataset, PromptInjectionDataset, AttributionGraphConverter]:
    """
    Create datasets from local directory with benign/ and injected/ subdirectories

    Args:
        dataset_path: Path to directory containing benign/ and injected/ subdirectories
        test_size: Fraction of data to use for testing
        val_size: Fraction of remaining data to use for validation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, converter)
    """

    from pathlib import Path

    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise ValueError(
            f"Dataset directory {dataset_path} does not exist or is not a directory")

    # Find benign and injected subdirectories
    benign_dir = dataset_dir / "benign"
    injected_dir = dataset_dir / "injected"

    if not benign_dir.exists():
        raise ValueError(f"Benign directory {benign_dir} not found")
    if not injected_dir.exists():
        raise ValueError(f"Injected directory {injected_dir} not found")

    # Collect all JSON files
    benign_files = list(benign_dir.glob("*.json"))
    injected_files = list(injected_dir.glob("*.json"))

    if not benign_files:
        raise ValueError(f"No JSON files found in {benign_dir}")
    if not injected_files:
        raise ValueError(f"No JSON files found in {injected_dir}")

    print(f"Loading from local directory: {dataset_path}")
    print(
        f"Found {len(benign_files)} benign and {len(injected_files)} injected files")

    # Convert to string paths and use existing function
    benign_file_paths = [str(f) for f in benign_files]
    injected_file_paths = [str(f) for f in injected_files]

    return create_datasets_from_converted_files(
        benign_files=benign_file_paths,
        injected_files=injected_file_paths,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )


def create_datasets_from_converted_files(benign_files, injected_files, test_size=0.2, val_size=0.1, random_state=42):
    """Create datasets from converted files that have JSON strings - memory efficient version"""

    print(
        f"Creating datasets from {len(benign_files)} benign and {len(injected_files)} injected converted files...")
    print("Using memory-efficient streaming approach...")

    # Initialize converter
    converter = AttributionGraphConverter()

    # PASS 1: Build vocabulary efficiently by streaming through files
    print("Pass 1: Building vocabulary from files...")

    def json_string_generator():
        """Generator that yields JSON strings without storing them all in memory"""
        all_files = [(f, 'benign') for f in benign_files] + \
            [(f, 'injected') for f in injected_files]

        successful_files = 0
        failed_files = 0

        for file_path, file_type in tqdm(all_files, desc="Reading files for vocabulary", unit="file"):
            try:
                # Check file size first - skip files that are too large or empty
                file_size = os.path.getsize(file_path)
                if file_size == 0:
                    failed_files += 1
                    continue
                # Skip files larger than 100MB (likely corrupted)
                elif file_size > 100 * 1024 * 1024:
                    tqdm.write(
                        f"Skipping large file {file_path}: {file_size / 1024 / 1024:.1f}MB")
                    failed_files += 1
                    continue

                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Handle both converted format and raw attribution graphs
                if 'json' in data:
                    # Converted format: {"json": "stringified_attribution_graph"}
                    yield data['json']
                    successful_files += 1
                elif 'nodes' in data and 'links' in data:
                    # Raw attribution graph format: {"nodes": [...], "links": [...]}
                    yield json.dumps(data)
                    successful_files += 1
                else:
                    # Unknown format
                    failed_files += 1
                    if failed_files <= 5:  # Only show first 5 format errors
                        tqdm.write(
                            f"Unknown file format in {file_path}: keys = {list(data.keys())}")

            except json.JSONDecodeError as e:
                failed_files += 1
                if failed_files <= 5:  # Only show first 5 JSON errors
                    tqdm.write(
                        f"JSON decode error in {file_path}: {str(e)[:100]}...")
            except Exception as e:
                failed_files += 1
                if failed_files <= 5:  # Only show first 5 other errors
                    tqdm.write(f"Error loading {file_path}: {str(e)[:100]}...")

        tqdm.write(
            f"Vocabulary building: {successful_files} successful, {failed_files} failed files")

    # Build vocabulary using generator (doesn't store all strings in memory)
    vocab_success = converter.build_vocabulary_from_json_strings(
        json_string_generator())

    if not vocab_success:
        raise ValueError("Failed to build vocabulary from converted files")

    # PASS 2: Convert to PyG Data objects (reuse the file reading, discard JSON strings immediately)
    print("Pass 2: Converting to PyG data objects...")
    all_data = []
    conversion_stats = {'benign': {'success': 0, 'failed': 0},
                        'injected': {'success': 0, 'failed': 0}}

    # Process benign files (label=0)
    for file_path in tqdm(benign_files, desc="Converting benign files", unit="file"):
        try:
            # Skip problematic files
            file_size = os.path.getsize(file_path)
            # Skip problematic files
            file_size = os.path.getsize(file_path)
            if file_size == 0 or file_size > 100 * 1024 * 1024:
                conversion_stats['benign']['failed'] += 1
                continue

            with open(file_path, 'r') as f:
                file_data = json.load(f)

            # Handle both formats
            json_string = None
            if 'json' in file_data:
                # Converted format
                json_string = file_data['json']
            elif 'nodes' in file_data and 'links' in file_data:
                # Raw attribution graph format
                json_string = json.dumps(file_data)

            if json_string:
                data = converter.json_string_to_pyg_data(json_string, label=0)
                if data is not None:
                    all_data.append(data)
                    conversion_stats['benign']['success'] += 1
                else:
                    conversion_stats['benign']['failed'] += 1
            else:
                conversion_stats['benign']['failed'] += 1

        except (json.JSONDecodeError, Exception) as e:
            conversion_stats['benign']['failed'] += 1

    # Process injected files (label=1)
    for file_path in tqdm(injected_files, desc="Converting injected files", unit="file"):
        try:

            # Skip problematic files
            file_size = os.path.getsize(file_path)
            if file_size == 0 or file_size > 100 * 1024 * 1024:
                conversion_stats['injected']['failed'] += 1
                continue

            with open(file_path, 'r') as f:
                file_data = json.load(f)

            # Handle both formats
            json_string = None
            if 'json' in file_data:
                # Converted format
                json_string = file_data['json']
            elif 'nodes' in file_data and 'links' in file_data:
                # Raw attribution graph format
                json_string = json.dumps(file_data)

            if json_string:
                data = converter.json_string_to_pyg_data(json_string, label=1)
                if data is not None:
                    all_data.append(data)
                    conversion_stats['injected']['success'] += 1
                else:
                    conversion_stats['injected']['failed'] += 1
            else:
                conversion_stats['injected']['failed'] += 1

        except (json.JSONDecodeError, Exception) as e:
            conversion_stats['injected']['failed'] += 1

    print(f"Conversion results:")
    print(
        f"  Benign graphs: {conversion_stats['benign']['success']} success, {conversion_stats['benign']['failed']} failed")
    print(
        f"  Injected graphs: {conversion_stats['injected']['success']} success, {conversion_stats['injected']['failed']} failed")

    if len(all_data) == 0:
        raise ValueError("No graphs were successfully converted!")

    # Create labels for stratification
    labels = [data.y.item() for data in all_data]

    # Check class balance
    num_benign = sum(1 for label in labels if label == 0)
    num_injected = sum(1 for label in labels if label == 1)

    print(
        f"Dataset composition: {num_benign} benign, {num_injected} injected graphs")

    if num_benign == 0 or num_injected == 0:
        raise ValueError(
            "Dataset must contain both benign and injected graphs!")

    # Stratified splits to maintain class balance
    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        all_data, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    train_data, val_data, _, _ = train_test_split(
        train_val_data, train_val_labels,
        test_size=val_size / (1 - test_size),
        stratify=train_val_labels,
        random_state=random_state
    )
    # Create datasets
    train_dataset = PromptInjectionDataset(train_data)
    val_dataset = PromptInjectionDataset(val_data)
    test_dataset = PromptInjectionDataset(test_data)

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
