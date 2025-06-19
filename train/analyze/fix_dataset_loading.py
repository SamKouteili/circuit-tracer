"""
Fixed dataset loading that handles variable-length arrays properly
"""

from datasets import load_dataset, Features, Value, Sequence
import json

def create_datasets_from_huggingface_fixed(
    dataset_name: str = "samkouteili/injection-attribution-graphs-small",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
):
    """
    Load HF dataset with explicit schema to handle variable-length arrays
    """
    
    print(f"Loading dataset '{dataset_name}' with fixed schema...")
    
    # Define explicit features schema to handle variable-length arrays
    features = Features({
        'metadata': {
            'slug': Value('string'),
            'scan': Value('string'),
            'transcoder_list': Sequence(Value('string')),  # Variable length
            'prompt_tokens': Sequence(Value('string')),    # Variable length
            'prompt': Value('string'),
            'node_threshold': Value('float64')
        },
        'qParams': {
            'nodeThreshold': Value('string'),
            'edgeThreshold': Value('string'),
            'maxNodes': Value('string'),
            'maxEdges': Value('string'),
            'displayTypes': Sequence(Value('string'))      # Variable length
        },
        'nodes': Sequence({                                # Variable length array of objects
            'node_id': Value('string'),
            'influence': Value('float64'),
            'activation': Value('float64'),
            'layer': Value('int64'),
            'feature': Value('int64'),
            'ctx_idx': Value('int64'),
            'feature_type': Value('string'),
            'is_target_logit': Value('bool')
        }),
        'links': Sequence({                                # Variable length array of objects
            'source': Value('string'),
            'target': Value('string'),
            'weight': Value('float64')
        })
    })
    
    try:
        # Load with explicit features
        dataset = load_dataset(
            dataset_name,
            features=features
        )
        
        print(f"✅ Successfully loaded dataset with explicit schema!")
        print(f"Dataset splits: {list(dataset.keys())}")
        
        # Check if we have the expected format
        if 'train' in dataset:
            train_data = dataset['train']
            print(f"Loaded {len(train_data)} examples")
            
            # Show structure of first example
            first_example = train_data[0]
            print(f"Example structure:")
            print(f"  nodes: {len(first_example['nodes'])} items")
            print(f"  links: {len(first_example['links'])} items")
            print(f"  metadata: {list(first_example['metadata'].keys())}")
            
            return dataset
        else:
            raise ValueError(f"Expected 'train' split, found: {list(dataset.keys())}")
            
    except Exception as e:
        print(f"❌ Fixed loading failed: {e}")
        
        # Fallback: try loading without explicit schema but with ignore_verifications
        print("Trying fallback approach...")
        try:
            dataset = load_dataset(
                dataset_name,
                ignore_verifications=True
            )
            print(f"✅ Fallback successful!")
            return dataset
        except Exception as e2:
            print(f"❌ Fallback also failed: {e2}")
            raise

if __name__ == "__main__":
    dataset = create_datasets_from_huggingface_fixed()