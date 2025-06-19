"""
Check the structure of the large dataset without downloading everything
"""

from datasets import load_dataset

def check_large_dataset_structure():
    print("Checking large dataset structure...")
    
    try:
        # Load just a few examples to check structure
        dataset = load_dataset(
            "samkouteili/injection-attribution-graphs", 
            streaming=True  # This prevents downloading the entire dataset
        )
        
        print(f"Dataset splits available: {list(dataset.keys())}")
        
        # Check each split
        for split_name in dataset.keys():
            print(f"\n=== {split_name.upper()} SPLIT ===")
            split_data = dataset[split_name]
            
            # Get first example
            first_example = next(iter(split_data))
            print(f"Keys in first example: {list(first_example.keys())}")
            
            # Check if it has the expected 'json' field
            if 'json' in first_example:
                json_content = first_example['json']
                print(f"✅ Has 'json' field (type: {type(json_content)}, length: {len(json_content) if isinstance(json_content, str) else 'N/A'})")
                
                # Try to parse the JSON content
                if isinstance(json_content, str):
                    import json
                    try:
                        parsed = json.loads(json_content)
                        print(f"  JSON content keys: {list(parsed.keys())}")
                        if 'nodes' in parsed:
                            print(f"  Number of nodes: {len(parsed['nodes'])}")
                        if 'links' in parsed:
                            print(f"  Number of links: {len(parsed['links'])}")
                    except Exception as e:
                        print(f"  ❌ Error parsing JSON: {e}")
                else:
                    print(f"  ❌ JSON field is not a string: {type(json_content)}")
            else:
                print(f"❌ No 'json' field found")
                
                # Check if it has nodes/links directly
                if 'nodes' in first_example:
                    print(f"  Has 'nodes' field directly (type: {type(first_example['nodes'])})")
                if 'links' in first_example:
                    print(f"  Has 'links' field directly (type: {type(first_example['links'])})")
                if 'metadata' in first_example:
                    print(f"  Has 'metadata' field (type: {type(first_example['metadata'])})")
            
            # Check a few more examples to see if structure is consistent
            print(f"\nChecking consistency across first 3 examples...")
            example_keys = []
            count = 0
            for example in split_data:
                example_keys.append(set(example.keys()))
                count += 1
                if count >= 3:
                    break
            
            # Check if all examples have same keys
            if len(set(frozenset(keys) for keys in example_keys)) == 1:
                print(f"✅ First {count} examples have consistent structure")
            else:
                print(f"❌ Inconsistent structure detected:")
                for i, keys in enumerate(example_keys):
                    print(f"  Example {i}: {sorted(keys)}")
                    
    except Exception as e:
        print(f"❌ Error checking dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_large_dataset_structure()