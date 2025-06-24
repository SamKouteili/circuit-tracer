#!/usr/bin/env python3
"""
Test script to verify all imports work before running training
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

print("Testing imports...")

try:
    import torch
    print("✅ torch imported successfully")
except ImportError as e:
    print(f"❌ torch import failed: {e}")
    sys.exit(1)

try:
    import torch_geometric
    print("✅ torch_geometric imported successfully")
except ImportError as e:
    print(f"❌ torch_geometric import failed: {e}")
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split
    print("✅ sklearn imported successfully")
except ImportError as e:
    print(f"❌ sklearn import failed: {e}")
    sys.exit(1)

try:
    from data_converter import AttributionGraphConverter
    print("✅ data_converter imported successfully")
except ImportError as e:
    print(f"❌ data_converter import failed: {e}")
    sys.exit(1)

try:
    from convert_and_load_dataset import download_and_convert_dataset
    print("✅ convert_and_load_dataset imported successfully")
except ImportError as e:
    print(f"❌ convert_and_load_dataset import failed: {e}")
    sys.exit(1)

try:
    from models import PromptInjectionGraphGPS
    print("✅ models imported successfully")
except ImportError as e:
    print(f"❌ models import failed: {e}")
    sys.exit(1)

try:
    from dataset import create_datasets_from_huggingface, create_data_loaders
    print("✅ dataset imported successfully")
except ImportError as e:
    print(f"❌ dataset import failed: {e}")
    sys.exit(1)

print("\n🎉 All imports successful! Training script should work.")