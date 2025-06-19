"""
Analyze the schema inconsistencies in the small dataset files
"""

import json
import os
from pathlib import Path

def analyze_file_schemas():
    data_dir = Path("/Users/samkouteili/rose/circuits/circuit-tracer/data_small")
    
    all_files = []
    all_files.extend(data_dir.glob("benign/*.json"))
    all_files.extend(data_dir.glob("injected/*.json"))
    
    print(f"Analyzing {len(all_files)} files...")
    
    schemas = {}
    field_types = {}
    
    for i, file_path in enumerate(all_files):
        print(f"\n=== File {i}: {file_path.name} ===")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Get top-level keys
            keys = set(data.keys())
            print(f"Top-level keys: {sorted(keys)}")
            
            # Track schema
            schema_key = tuple(sorted(keys))
            if schema_key not in schemas:
                schemas[schema_key] = []
            schemas[schema_key].append(file_path.name)
            
            # Check field types
            for key, value in data.items():
                if key not in field_types:
                    field_types[key] = {}
                
                value_type = type(value).__name__
                if isinstance(value, list) and len(value) > 0:
                    value_type = f"list[{type(value[0]).__name__}]"
                elif isinstance(value, dict) and len(value) > 0:
                    # Check if it's a dict with consistent value types
                    dict_value_types = set(type(v).__name__ for v in value.values())
                    if len(dict_value_types) == 1:
                        value_type = f"dict[{list(dict_value_types)[0]}]"
                    else:
                        value_type = f"dict[mixed: {dict_value_types}]"
                
                if value_type not in field_types[key]:
                    field_types[key][value_type] = []
                field_types[key][value_type].append(file_path.name)
                
                print(f"  {key}: {value_type}")
                
        except Exception as e:
            print(f"❌ Error reading {file_path.name}: {e}")
    
    print(f"\n{'='*50}")
    print("SCHEMA SUMMARY")
    print(f"{'='*50}")
    
    print(f"\nFound {len(schemas)} different schemas:")
    for i, (schema, files) in enumerate(schemas.items()):
        print(f"\nSchema {i+1}: {schema}")
        print(f"  Files: {files}")
    
    print(f"\nFIELD TYPE INCONSISTENCIES:")
    for field, types in field_types.items():
        if len(types) > 1:
            print(f"\n❌ Field '{field}' has inconsistent types:")
            for type_name, files in types.items():
                print(f"  {type_name}: {files}")
        else:
            type_name = list(types.keys())[0]
            print(f"✅ Field '{field}': consistent {type_name}")

if __name__ == "__main__":
    analyze_file_schemas()