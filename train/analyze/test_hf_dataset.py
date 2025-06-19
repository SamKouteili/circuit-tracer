"""
Test loading the Hugging Face dataset
"""

from datasets import load_dataset


def test_hf_dataset():
    """Test loading the samkouteili/injection-attribution-graphs dataset"""
    print("Testing Hugging Face dataset loading...")

    try:
        dataset = load_dataset("samkouteili/injection-attribution-graphs-tiny")

        print(f"✅ Successfully loaded dataset!")
        print(f"Dataset splits: {list(dataset.keys())}")

        if 'benign' in dataset:
            print(f"Benign examples: {len(dataset['benign'])}")
            # Check first example structure
            first_benign = dataset['benign'][0]
            print(f"Benign example keys: {list(first_benign.keys())}")
            if 'json' in first_benign:
                json_str = first_benign['json']
                print(f"JSON string length: {len(json_str)} characters")
                # Try to parse it
                import json
                try:
                    parsed = json.loads(json_str)
                    print(f"JSON structure keys: {list(parsed.keys())}")
                    if 'nodes' in parsed:
                        print(f"Number of nodes: {len(parsed['nodes'])}")
                    if 'links' in parsed:
                        print(f"Number of links: {len(parsed['links'])}")
                except Exception as e:
                    print(f"Error parsing JSON: {e}")

        if 'injected' in dataset:
            print(f"Injected examples: {len(dataset['injected'])}")

        return True

    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False


if __name__ == "__main__":
    test_hf_dataset()
