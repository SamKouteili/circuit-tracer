"""Debug the dataset structure"""

from datasets import load_dataset

def check_dataset_structure():
    try:
        dataset = load_dataset("samkouteili/injection-attribution-graphs-tiny")
        print(f"Dataset splits available: {list(dataset.keys())}")
        
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            print(f"\nSplit '{split_name}':")
            print(f"  Number of examples: {len(split_data)}")
            if len(split_data) > 0:
                first_item = split_data[0]
                print(f"  Keys in first item: {list(first_item.keys())}")
                
                # Check metadata for labeling info
                if 'metadata' in first_item:
                    print(f"  Metadata: {first_item['metadata']}")
                
                # Show structure of a few examples
                for i in range(min(3, len(split_data))):
                    item = split_data[i]
                    print(f"  Example {i}:")
                    if 'metadata' in item:
                        print(f"    Metadata: {item['metadata']}")
                    if 'nodes' in item:
                        print(f"    Number of nodes: {len(item['nodes'])}")
                    if 'links' in item:
                        print(f"    Number of links: {len(item['links'])}")
                
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    check_dataset_structure()