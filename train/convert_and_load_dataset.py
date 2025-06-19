"""
Download and convert HuggingFace dataset files locally to the expected format
"""

import json
import os
from pathlib import Path
import requests
from huggingface_hub import hf_hub_download, list_repo_files
from typing import List, Tuple
import tempfile
import shutil

def download_dataset_files(dataset_name: str, local_dir: str) -> None:
    """Download all JSON files from HuggingFace dataset"""
    
    print(f"Downloading files from {dataset_name}...")
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    # List all files in the repo
    try:
        files = list_repo_files(dataset_name, repo_type="dataset")
        json_files = [f for f in files if f.endswith('.json')]
        
        print(f"Found {len(json_files)} JSON files to download:")
        for file in json_files:
            print(f"  - {file}")
        
        # Download each file
        for file_path in json_files:
            print(f"Downloading {file_path}...")
            local_file_path = local_path / file_path
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            downloaded_path = hf_hub_download(
                repo_id=dataset_name,
                filename=file_path,
                repo_type="dataset",
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            print(f"  -> {downloaded_path}")
            
    except Exception as e:
        print(f"Error downloading files: {e}")
        raise

def convert_files_to_expected_format(source_dir: str, output_dir: str) -> Tuple[List[str], List[str]]:
    """Convert downloaded files to expected format with 'json' field"""
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    benign_files = []
    injected_files = []
    
    print(f"Converting files from {source_dir} to {output_dir}...")
    
    # Process benign files
    benign_dir = source_path / "benign"
    if benign_dir.exists():
        output_benign_dir = output_path / "benign"
        output_benign_dir.mkdir(exist_ok=True)
        
        for json_file in benign_dir.glob("*.json"):
            print(f"Converting benign file: {json_file.name}")
            
            with open(json_file, 'r') as f:
                original_data = json.load(f)
            
            # Extract just the nodes and links for the JSON string
            attribution_graph = {
                "nodes": original_data.get("nodes", []),
                "links": original_data.get("links", [])
            }
            
            # Create the expected format
            converted_data = {
                "json": json.dumps(attribution_graph)
            }
            
            # Save converted file
            output_file = output_benign_dir / json_file.name
            with open(output_file, 'w') as f:
                json.dump(converted_data, f, indent=2)
            
            benign_files.append(str(output_file))
    
    # Process injected files
    injected_dir = source_path / "injected"
    if injected_dir.exists():
        output_injected_dir = output_path / "injected"
        output_injected_dir.mkdir(exist_ok=True)
        
        for json_file in injected_dir.glob("*.json"):
            print(f"Converting injected file: {json_file.name}")
            
            with open(json_file, 'r') as f:
                original_data = json.load(f)
            
            # Extract just the nodes and links for the JSON string
            attribution_graph = {
                "nodes": original_data.get("nodes", []),
                "links": original_data.get("links", [])
            }
            
            # Create the expected format
            converted_data = {
                "json": json.dumps(attribution_graph)
            }
            
            # Save converted file
            output_file = output_injected_dir / json_file.name
            with open(output_file, 'w') as f:
                json.dump(converted_data, f, indent=2)
            
            injected_files.append(str(output_file))
    
    print(f"✅ Conversion complete!")
    print(f"  Benign files: {len(benign_files)}")
    print(f"  Injected files: {len(injected_files)}")
    
    return benign_files, injected_files

def download_and_convert_dataset(
    dataset_name: str = "samkouteili/injection-attribution-graphs-small",
    cache_dir: str = None
) -> Tuple[List[str], List[str]]:
    """
    Download and convert HuggingFace dataset to expected format
    
    Returns:
        Tuple of (benign_files, injected_files) paths
    """
    
    if cache_dir is None:
        cache_dir = f"./converted_datasets/{dataset_name.replace('/', '_')}"
    
    cache_path = Path(cache_dir)
    
    # Check if already converted
    converted_benign = list((cache_path / "converted" / "benign").glob("*.json")) if (cache_path / "converted" / "benign").exists() else []
    converted_injected = list((cache_path / "converted" / "injected").glob("*.json")) if (cache_path / "converted" / "injected").exists() else []
    
    if converted_benign or converted_injected:
        print(f"✅ Found already converted files in {cache_dir}")
        return [str(f) for f in converted_benign], [str(f) for f in converted_injected]
    
    # Download raw files
    raw_dir = cache_path / "raw"
    download_dataset_files(dataset_name, str(raw_dir))
    
    # Convert files
    converted_dir = cache_path / "converted"
    benign_files, injected_files = convert_files_to_expected_format(str(raw_dir), str(converted_dir))
    
    return benign_files, injected_files

def test_conversion():
    """Test the conversion process"""
    print("Testing dataset download and conversion...")
    
    try:
        benign_files, injected_files = download_and_convert_dataset(
            "samkouteili/injection-attribution-graphs-small"
        )
        
        print(f"\n✅ Successfully converted dataset!")
        print(f"Benign files: {len(benign_files)}")
        print(f"Injected files: {len(injected_files)}")
        
        # Test loading one converted file
        if benign_files:
            print(f"\nTesting converted file format...")
            with open(benign_files[0], 'r') as f:
                test_data = json.load(f)
            
            print(f"Converted file keys: {list(test_data.keys())}")
            if 'json' in test_data:
                json_content = json.loads(test_data['json'])
                print(f"JSON content keys: {list(json_content.keys())}")
                if 'nodes' in json_content:
                    print(f"Number of nodes: {len(json_content['nodes'])}")
                if 'links' in json_content:
                    print(f"Number of links: {len(json_content['links'])}")
                print("✅ Format looks correct!")
            
        return benign_files, injected_files
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return [], []

if __name__ == "__main__":
    test_conversion()