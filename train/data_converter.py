"""
Data converter for attribution graphs to PyTorch Geometric format
"""

import torch
import json
import numpy as np
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Optional
import os

class AttributionGraphConverter:
    """Converts JSON attribution graphs to PyTorch Geometric Data objects"""
    
    def __init__(self):
        # Track all possible node types across dataset for consistent encoding
        self.node_vocab = {}  # maps node_id -> integer index
        self.feature_names = [
            'influence', 'activation', 'layer', 'ctx_idx', 'feature',
            'is_cross_layer_transcoder', 'is_mlp_error', 'is_embedding', 'is_target_logit'
        ]
        self.feature_dims = len(self.feature_names)
        
    def build_vocabulary(self, graph_files: List[str]):
        """First pass: build vocabulary of all possible nodes across all graphs"""
        all_nodes = set()
        
        print(f"Building vocabulary from {len(graph_files)} files...")
        
        for file_path in graph_files:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} not found, skipping...")
                continue
                
            try:
                with open(file_path, 'r') as f:
                    graph_data = json.load(f)
                
                if 'nodes' not in graph_data:
                    print(f"Warning: No 'nodes' key in {file_path}, skipping...")
                    continue
                
                for node in graph_data['nodes']:
                    if 'node_id' in node and not self._is_error_node(node):
                        node_id = node['node_id']
                        all_nodes.add(node_id)
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Create mapping: node_id -> integer index
        self.node_vocab = {node_id: idx for idx, node_id in enumerate(sorted(all_nodes))}
        print(f"Built vocabulary with {len(self.node_vocab)} unique nodes")
        
        return len(self.node_vocab) > 0
    
    def build_vocabulary_from_json_strings(self, json_strings: List[str]):
        """Build vocabulary from list of JSON strings"""
        all_nodes = set()
        
        print(f"Building vocabulary from {len(json_strings)} JSON strings...")
        
        for i, json_string in enumerate(json_strings):
            try:
                graph_data = json.loads(json_string)
                
                if 'nodes' not in graph_data:
                    print(f"Warning: No 'nodes' key in JSON string {i}, skipping...")
                    continue
                
                for node in graph_data['nodes']:
                    if 'node_id' in node and not self._is_error_node(node):
                        node_id = node['node_id']
                        all_nodes.add(node_id)
                        
            except Exception as e:
                print(f"Error processing JSON string {i}: {e}")
                continue
        
        # Create mapping: node_id -> integer index
        self.node_vocab = {node_id: idx for idx, node_id in enumerate(sorted(all_nodes))}
        print(f"Built vocabulary with {len(self.node_vocab)} unique nodes")
        
        return len(self.node_vocab) > 0
        
    def extract_node_features(self, node: Dict) -> List[float]:
        """Extract numeric features from a node"""
        features = []
        
        # Basic numeric features - handle None values
        influence = node.get('influence', 0.0)
        features.append(float(influence if influence is not None else 0.0))
        
        activation = node.get('activation', 0.0)
        features.append(float(activation if activation is not None else 0.0))
        
        layer = node.get('layer', 0)
        features.append(float(layer if layer is not None else 0))
        
        ctx_idx = node.get('ctx_idx', 0)
        features.append(float(ctx_idx if ctx_idx is not None else 0))
        
        feature = node.get('feature', 0)
        features.append(float(feature if feature is not None else 0))
        
        # One-hot encoded feature types
        feature_type = node.get('feature_type', 'unknown')
        features.append(1.0 if feature_type == 'cross layer transcoder' else 0.0)
        features.append(1.0 if feature_type == 'mlp reconstruction error' else 0.0)
        features.append(1.0 if feature_type == 'embedding' else 0.0)
        features.append(1.0 if node.get('is_target_logit', False) else 0.0)
        
        return features
        
    def json_string_to_pyg_data(self, json_string: str, label: int) -> Optional[Data]:
        """Convert JSON string to PyTorch Geometric Data object"""
        
        try:
            graph_data = json.loads(json_string)
        except Exception as e:
            print(f"Error parsing JSON string: {e}")
            return None
        
        # Validate structure
        if 'nodes' not in graph_data or 'links' not in graph_data:
            print(f"Error: Invalid graph structure in JSON string")
            return None
        
        nodes = graph_data['nodes']
        links = graph_data['links']
        
        if len(nodes) == 0:
            print(f"Warning: No nodes in JSON string")
            return None
        
        # Extract nodes and create feature matrix (skip error nodes)
        node_features = []
        node_mapping = {}  # map node_id to position in this graph
        valid_nodes = []
        
        for node in nodes:
            if 'node_id' not in node or self._is_error_node(node):
                continue
                
            valid_nodes.append(node)
        
        for i, node in enumerate(valid_nodes):
            node_id = node['node_id']
            node_mapping[node_id] = i
            
            # Extract numeric features
            features = self.extract_node_features(node)
            node_features.append(features)
        
        if len(node_features) == 0:
            print(f"Warning: No valid nodes found in JSON string")
            return None
        
        # Convert to tensor
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Extract edges
        edge_indices = []
        edge_weights = []
        
        for link in links:
            if 'source' not in link or 'target' not in link:
                continue
                
            source_id = link['source']
            target_id = link['target']
            
            # Only include edges where both nodes exist in our node set
            if source_id in node_mapping and target_id in node_mapping:
                source_idx = node_mapping[source_id]
                target_idx = node_mapping[target_id]
                
                edge_indices.append([source_idx, target_idx])
                weight = link.get('weight', 1.0)
                edge_weights.append(float(weight if weight is not None else 1.0))
        
        # Convert edges to tensor format [2, num_edges]
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
        else:
            # Create empty edge tensors for graphs with no edges
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        
        # Create PyG Data object
        data = Data(
            x=x,                    # Node features [num_nodes, num_features]
            edge_index=edge_index,  # Edge connectivity [2, num_edges]
            edge_attr=edge_attr,    # Edge weights [num_edges, 1]
            y=torch.tensor([label], dtype=torch.long),  # Graph label
            num_nodes=len(node_features)
        )
        
        return data
    
    def json_to_pyg_data(self, json_file: str, label: int) -> Optional[Data]:
        """Convert single JSON attribution graph file to PyTorch Geometric Data object"""
        
        if not os.path.exists(json_file):
            print(f"Error: File {json_file} not found")
            return None
            
        try:
            with open(json_file, 'r') as f:
                json_string = f.read()
        except Exception as e:
            print(f"Error reading file {json_file}: {e}")
            return None
        
        return self.json_string_to_pyg_data(json_string, label)
    
    def _is_error_node(self, node: Dict) -> bool:
        """Check if a node is an error node that should be pruned"""
        node_id = node.get('node_id', '')
        feature_type = node.get('feature_type', '')
        
        # Check for error node ID patterns
        if node_id.startswith('E_'):
            return True
            
        # Check for error feature types
        if feature_type in ['mlp reconstruction error', 'embedding']:
            return True
            
        return False
    
    def get_feature_dim(self) -> int:
        """Get the dimension of node features"""
        return self.feature_dims

def test_converter():
    """Test the converter with the example JSON files"""
    print("Testing AttributionGraphConverter...")
    
    # Get paths to JSON files
    base_path = "/Users/samkouteili/rose/circuits"
    json_files = [
        os.path.join(base_path, "Ellen Pompeo Attribution Graph.json"),
        os.path.join(base_path, "Ellen Pompeo Prompt Injection.json"),
        os.path.join(base_path, "Prompt Injection Cyfyre.json")
    ]
    
    # Initialize converter
    converter = AttributionGraphConverter()
    
    # Build vocabulary
    vocab_success = converter.build_vocabulary(json_files)
    if not vocab_success:
        print("Failed to build vocabulary!")
        return False
    
    print(f"Feature dimension: {converter.get_feature_dim()}")
    print(f"Feature names: {converter.feature_names}")
    
    # Test conversion of each file
    test_results = []
    labels = [0, 1, 1]  # normal, injection, injection
    
    for i, (json_file, label) in enumerate(zip(json_files, labels)):
        print(f"\nTesting file {i+1}: {os.path.basename(json_file)}")
        
        data = converter.json_to_pyg_data(json_file, label)
        
        if data is None:
            print(f"  Failed to convert {json_file}")
            test_results.append(False)
            continue
        
        print(f"  Success! Data shape:")
        print(f"    Nodes: {data.num_nodes}")
        print(f"    Node features: {data.x.shape}")
        print(f"    Edges: {data.edge_index.shape[1]}")
        print(f"    Edge attributes: {data.edge_attr.shape}")
        print(f"    Label: {data.y.item()}")
        
        # Basic validation
        if data.x.shape[1] != converter.get_feature_dim():
            print(f"    ERROR: Feature dimension mismatch!")
            test_results.append(False)
            continue
        
        if data.edge_index.shape[0] != 2:
            print(f"    ERROR: Edge index shape wrong!")
            test_results.append(False)
            continue
            
        print(f"    Validation: PASSED")
        test_results.append(True)
    
    success_count = sum(test_results)
    total_count = len(test_results)
    
    print(f"\n=== CONVERSION TEST RESULTS ===")
    print(f"Successfully converted: {success_count}/{total_count} files")
    
    if success_count > 0:
        print("✅ Data conversion pipeline is working!")
        return True
    else:
        print("❌ Data conversion failed!")
        return False

if __name__ == "__main__":
    test_converter()