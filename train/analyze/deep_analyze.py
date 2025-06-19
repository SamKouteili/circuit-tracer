"""
Deep analysis of field value types across files
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def deep_analyze_fields():
    data_dir = Path("/Users/samkouteili/rose/circuits/circuit-tracer/data_small")
    
    all_files = []
    all_files.extend(data_dir.glob("benign/*.json"))
    all_files.extend(data_dir.glob("injected/*.json"))
    
    print(f"Deep analyzing {len(all_files)} files...")
    
    # Track all field paths and their types
    field_types = defaultdict(lambda: defaultdict(list))
    
    def analyze_value(value, path=""):
        """Recursively analyze value types"""
        value_type = type(value).__name__
        
        if isinstance(value, dict):
            field_types[path][value_type].append("dict_container")
            for key, val in value.items():
                new_path = f"{path}.{key}" if path else key
                analyze_value(val, new_path)
        elif isinstance(value, list):
            field_types[path][f"list[{len(value)}]"].append("list_container")
            if value:  # Non-empty list
                # Check if all items have same type
                item_types = set(type(item).__name__ for item in value)
                if len(item_types) == 1:
                    field_types[path][f"list[{list(item_types)[0]}]"].append("consistent_list")
                else:
                    field_types[path][f"list[mixed:{item_types}]"].append("mixed_list")
                
                # Analyze first few items if they're complex
                for i, item in enumerate(value[:3]):
                    if isinstance(item, (dict, list)):
                        analyze_value(item, f"{path}[{i}]")
        else:
            field_types[path][value_type].append(str(value)[:50] if len(str(value)) > 50 else str(value))
    
    for i, file_path in enumerate(all_files[:5]):  # Analyze first 5 files
        print(f"\nAnalyzing file: {file_path.name}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            analyze_value(data)
                
        except Exception as e:
            print(f"❌ Error reading {file_path.name}: {e}")
    
    print(f"\n{'='*60}")
    print("DETAILED FIELD TYPE ANALYSIS")
    print(f"{'='*60}")
    
    # Look for inconsistencies
    for field_path, type_info in sorted(field_types.items()):
        if len(type_info) > 1:
            print(f"\n❌ INCONSISTENT: {field_path}")
            for type_name, examples in type_info.items():
                print(f"  {type_name}: {examples[:3]}...")  # Show first 3 examples
        elif any(key.startswith('list[mixed:') or key.startswith('dict[mixed:') for key in type_info.keys()):
            print(f"\n⚠️  MIXED TYPES: {field_path}")
            for type_name, examples in type_info.items():
                print(f"  {type_name}: {examples[:3]}...")
    
    print(f"\n{'='*40}")
    print("SUMMARY")
    print(f"{'='*40}")
    
    problematic_fields = [path for path, types in field_types.items() if len(types) > 1]
    if problematic_fields:
        print(f"❌ Found {len(problematic_fields)} fields with inconsistent types:")
        for field in problematic_fields:
            print(f"  - {field}")
    else:
        print("✅ No obviously inconsistent field types found")
        print("The PyArrow issue might be due to:")
        print("  1. Array length inconsistencies")
        print("  2. Nested object structure variations")
        print("  3. Special characters or encoding issues")

if __name__ == "__main__":
    deep_analyze_fields()