#!/usr/bin/env python3
"""
Script to find nodes present in prompt injection files but not in the baseline Ellen Pompeo graph.
Orders results by influence (total incoming + outgoing connections).
"""

import json
from typing import Dict, List, Tuple, Set

def load_graph(filepath: str) -> Dict:
    """Load attribution graph from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_nodes_with_influence(graph: Dict) -> Dict[str, float]:
    """
    Extract all nodes from graph and calculate their influence.
    Influence = sum of absolute values of all incoming and outgoing edge weights.
    """
    nodes_influence = {}
    
    # Get nodes from the graph
    if 'nodes' in graph:
        nodes = graph['nodes']
    else:
        print(f"Warning: No 'nodes' key found in graph. Available keys: {list(graph.keys())}")
        return {}
    
    # Initialize influence scores (excluding error nodes)
    for node in nodes:
        node_id = node.get('node_id', node.get('id', ''))
        # Skip error nodes (those starting with 'E_' or having feature_type 'mlp reconstruction error' or 'embedding')
        feature_type = node.get('feature_type', '')
        if (node_id.startswith('E_') or 
            feature_type == 'mlp reconstruction error' or 
            feature_type == 'embedding'):
            continue
        nodes_influence[node_id] = 0.0
    
    # Calculate influence from edges/links
    edges_key = 'links' if 'links' in graph else 'edges'
    if edges_key in graph:
        edges = graph[edges_key]
        for edge in edges:
            source = edge.get('source', '')
            target = edge.get('target', '')
            # Try different possible weight field names
            weight = edge.get('weight', edge.get('value', edge.get('strength', 0.0)))
            weight = abs(float(weight))
            
            # Add to source (outgoing) and target (incoming) influence
            if source in nodes_influence:
                nodes_influence[source] += weight
            if target in nodes_influence:
                nodes_influence[target] += weight
    else:
        print(f"Warning: No 'edges' or 'links' key found. Available keys: {list(graph.keys())}")
    
    return nodes_influence

def get_node_details(graph: Dict, node_id: str) -> Dict:
    """Get detailed information about a specific node."""
    if 'nodes' not in graph:
        return {}
    
    for node in graph['nodes']:
        if node.get('node_id', node.get('id', '')) == node_id:
            return node
    return {}

def main():
    # File paths
    baseline_file = "Ellen Pompeo Attribution Graph.json"
    injection1_file = "Ellen Pompeo Prompt Injection.json"
    injection2_file = "Prompt Injection Cyfyre.json"
    
    print("Loading attribution graphs...")
    
    try:
        # Load all three graphs
        baseline_graph = load_graph(baseline_file)
        injection1_graph = load_graph(injection1_file)
        injection2_graph = load_graph(injection2_file)
        
        print(f"Baseline graph loaded: {len(baseline_graph.get('nodes', []))} nodes")
        print(f"Injection 1 graph loaded: {len(injection1_graph.get('nodes', []))} nodes")
        print(f"Injection 2 graph loaded: {len(injection2_graph.get('nodes', []))} nodes")
        
        # Extract nodes and their influence scores
        baseline_nodes = extract_nodes_with_influence(baseline_graph)
        injection1_nodes = extract_nodes_with_influence(injection1_graph)
        injection2_nodes = extract_nodes_with_influence(injection2_graph)
        
        print(f"\nBaseline nodes with influence: {len(baseline_nodes)}")
        print(f"Injection 1 nodes with influence: {len(injection1_nodes)}")
        print(f"Injection 2 nodes with influence: {len(injection2_nodes)}")
        
        # Find nodes that are in both injection graphs but not in baseline
        baseline_node_ids = set(baseline_nodes.keys())
        injection1_node_ids = set(injection1_nodes.keys())
        injection2_node_ids = set(injection2_nodes.keys())
        
        # Nodes present in both injection graphs
        common_injection_nodes = injection1_node_ids.intersection(injection2_node_ids)
        
        # Nodes in both injection graphs but not in baseline
        unique_to_injections = common_injection_nodes - baseline_node_ids
        
        print(f"\nNodes in both injection graphs: {len(common_injection_nodes)}")
        print(f"Nodes unique to injection graphs (not in baseline): {len(unique_to_injections)}")
        
        if not unique_to_injections:
            print("No nodes found that are in both injection graphs but missing from baseline.")
            
            # Alternative: show nodes in either injection graph but not baseline
            either_injection = injection1_node_ids.union(injection2_node_ids)
            unique_to_either = either_injection - baseline_node_ids
            print(f"\nAlternative: Nodes in either injection graph but not baseline: {len(unique_to_either)}")
            
            if unique_to_either:
                # Calculate average influence across both injection graphs
                results = []
                for node_id in unique_to_either:
                    # Skip error nodes
                    if node_id.startswith('E_'):
                        continue
                    
                    # Get node details to check feature type
                    details1 = get_node_details(injection1_graph, node_id)
                    details2 = get_node_details(injection2_graph, node_id)
                    details = details1 if details1 else details2
                    
                    feature_type = details.get('feature_type', '')
                    if (feature_type == 'mlp reconstruction error' or 
                        feature_type == 'embedding'):
                        continue
                    
                    influence1 = injection1_nodes.get(node_id, 0.0)
                    influence2 = injection2_nodes.get(node_id, 0.0)
                    avg_influence = (influence1 + influence2) / 2.0
                    
                    results.append({
                        'id': node_id,
                        'influence': avg_influence,
                        'details': details,
                        'in_injection1': node_id in injection1_node_ids,
                        'in_injection2': node_id in injection2_node_ids
                    })
                
                # Sort by influence (descending)
                results.sort(key=lambda x: x['influence'], reverse=True)
                
                print(f"\nTop {min(20, len(results))} nodes unique to injection graphs (by influence):")
                print("-" * 80)
                for i, result in enumerate(results[:20]):
                    node_type = result['details'].get('type', 'unknown')
                    label = result['details'].get('label', 'no label')
                    in1 = "✓" if result['in_injection1'] else "✗"
                    in2 = "✓" if result['in_injection2'] else "✗"
                    
                    # Parse node ID for better understanding
                    node_parts = result['id'].split('_')
                    if len(node_parts) >= 3:
                        layer, feature, pos = node_parts[0], node_parts[1], node_parts[2]
                        node_desc = f"Layer {layer}, Feature {feature}, Position {pos}"
                    else:
                        node_desc = "Unknown format"
                    
                    # Get additional details
                    details = result['details']
                    feature_type = details.get('feature_type', 'unknown')
                    activation = details.get('activation', 0.0)
                    if activation is None:
                        activation = 0.0
                    
                    print(f"{i+1:2d}. {result['id']}")
                    print(f"    Description: {node_desc}")
                    print(f"    Influence: {result['influence']:.4f}")
                    print(f"    Feature Type: {feature_type}")
                    print(f"    Activation: {activation:.4f}")
                    print(f"    In Injection1: {in1}, In Injection2: {in2}")
                    if label != 'no label':
                        print(f"    Label: {label}")
                    print()
            
        else:
            # Calculate average influence for nodes in both injection graphs
            results = []
            for node_id in unique_to_injections:
                # Skip error nodes
                if node_id.startswith('E_'):
                    continue
                
                # Get node details to check feature type
                details1 = get_node_details(injection1_graph, node_id)
                details2 = get_node_details(injection2_graph, node_id)
                details = details1 if details1 else details2
                
                feature_type = details.get('feature_type', '')
                if (feature_type == 'mlp reconstruction error' or 
                    feature_type == 'embedding'):
                    continue
                
                influence1 = injection1_nodes[node_id]
                influence2 = injection2_nodes[node_id]
                avg_influence = (influence1 + influence2) / 2.0
                
                results.append({
                    'id': node_id,
                    'influence': avg_influence,
                    'details': details
                })
            
            # Sort by influence (descending)
            results.sort(key=lambda x: x['influence'], reverse=True)
            
            print(f"\nTop {min(20, len(results))} nodes unique to both injection graphs (by influence):")
            print("-" * 80)
            for i, result in enumerate(results[:20]):
                node_type = result['details'].get('type', 'unknown')
                label = result['details'].get('label', 'no label')
                
                # Parse node ID for better understanding
                node_parts = result['id'].split('_')
                if len(node_parts) >= 3:
                    layer, feature, pos = node_parts[0], node_parts[1], node_parts[2]
                    node_desc = f"Layer {layer}, Feature {feature}, Position {pos}"
                else:
                    node_desc = "Unknown format"
                
                # Get additional details
                details = result['details']
                feature_type = details.get('feature_type', 'unknown')
                activation = details.get('activation', 0.0)
                if activation is None:
                    activation = 0.0
                
                print(f"{i+1:2d}. {result['id']}")
                print(f"    Description: {node_desc}")
                print(f"    Influence: {result['influence']:.4f}")
                print(f"    Feature Type: {feature_type}")
                print(f"    Activation: {activation:.4f}")
                if label != 'no label':
                    print(f"    Label: {label}")
                print()
    
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()