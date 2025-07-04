"""
Utility functions for explanation processing and analysis
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from torch_geometric.data import Data, Batch
import pickle
import json
import os
from pathlib import Path
import time
from datetime import datetime

from explanation import AttributionGraphExplanation, ExplanationBatch


class ExplanationCache:
    """
    Cache for storing and retrieving explanations to avoid recomputation
    """
    
    def __init__(self, cache_dir: str = "./explanation_cache"):
        """
        Initialize explanation cache
        
        Args:
            cache_dir: Directory to store cached explanations
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Metadata file to track cache contents
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"explanations": {}, "created": datetime.now().isoformat()}
    
    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _get_cache_key(self, graph_id: str, model_hash: str, config_hash: str) -> str:
        """Generate cache key for explanation"""
        return f"{graph_id}_{model_hash}_{config_hash}"
    
    def _get_model_hash(self, model) -> str:
        """Get hash of model parameters for cache key"""
        # Simple hash based on parameter count and first few parameters
        total_params = sum(p.numel() for p in model.parameters())
        param_sample = []
        for i, p in enumerate(model.parameters()):
            if i < 3:  # Sample first 3 parameters
                param_sample.extend(p.flatten()[:10].tolist())
        param_hash = hash(tuple([total_params] + param_sample))
        return str(abs(param_hash))[:8]
    
    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        """Get hash of explanation configuration"""
        config_str = json.dumps(config, sort_keys=True)
        return str(abs(hash(config_str)))[:8]
    
    def has_explanation(self, graph_id: str, model, config: Dict[str, Any]) -> bool:
        """Check if explanation exists in cache"""
        model_hash = self._get_model_hash(model)
        config_hash = self._get_config_hash(config)
        cache_key = self._get_cache_key(graph_id, model_hash, config_hash)
        
        return cache_key in self.metadata["explanations"]
    
    def save_explanation(self, 
                        explanation: AttributionGraphExplanation,
                        model,
                        config: Dict[str, Any]):
        """Save explanation to cache"""
        model_hash = self._get_model_hash(model)
        config_hash = self._get_config_hash(config)
        cache_key = self._get_cache_key(explanation.graph_id, model_hash, config_hash)
        
        # Save explanation to file
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(explanation, f)
        
        # Update metadata
        self.metadata["explanations"][cache_key] = {
            "graph_id": explanation.graph_id,
            "model_hash": model_hash,
            "config_hash": config_hash,
            "file_path": str(cache_file),
            "created": datetime.now().isoformat(),
            "true_label": explanation.true_label,
            "predicted_label": explanation.predicted_label
        }
        
        self._save_metadata()
        print(f"💾 Cached explanation for {explanation.graph_id}")
    
    def load_explanation(self,
                        graph_id: str,
                        model,
                        config: Dict[str, Any]) -> Optional[AttributionGraphExplanation]:
        """Load explanation from cache"""
        model_hash = self._get_model_hash(model)
        config_hash = self._get_config_hash(config)
        cache_key = self._get_cache_key(graph_id, model_hash, config_hash)
        
        if cache_key not in self.metadata["explanations"]:
            return None
        
        cache_file = Path(self.metadata["explanations"][cache_key]["file_path"])
        
        if not cache_file.exists():
            # Clean up metadata for missing file
            del self.metadata["explanations"][cache_key]
            self._save_metadata()
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                explanation = pickle.load(f)
            print(f"📁 Loaded cached explanation for {graph_id}")
            return explanation
        except Exception as e:
            print(f"⚠️ Failed to load cached explanation for {graph_id}: {e}")
            return None
    
    def clear_cache(self):
        """Clear all cached explanations"""
        for cache_key, info in self.metadata["explanations"].items():
            cache_file = Path(info["file_path"])
            if cache_file.exists():
                cache_file.unlink()
        
        self.metadata = {"explanations": {}, "created": datetime.now().isoformat()}
        self._save_metadata()
        print("🗑️ Cleared explanation cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        num_explanations = len(self.metadata["explanations"])
        
        # Calculate total cache size
        total_size = 0
        for info in self.metadata["explanations"].values():
            cache_file = Path(info["file_path"])
            if cache_file.exists():
                total_size += cache_file.stat().st_size
        
        # Count by label
        label_counts = {"benign": 0, "injected": 0}
        for info in self.metadata["explanations"].values():
            if info["true_label"] == 0:
                label_counts["benign"] += 1
            else:
                label_counts["injected"] += 1
        
        return {
            "num_explanations": num_explanations,
            "total_size_mb": total_size / (1024 * 1024),
            "label_distribution": label_counts,
            "cache_directory": str(self.cache_dir),
            "created": self.metadata.get("created", "unknown")
        }


class BatchProcessor:
    """
    Utility for processing explanations in batches with progress tracking
    """
    
    def __init__(self, 
                 explainer,
                 cache: Optional[ExplanationCache] = None,
                 batch_size: int = 32,
                 max_workers: Optional[int] = None):
        """
        Initialize batch processor
        
        Args:
            explainer: CircuitTracerGNNExplainer instance
            cache: Optional explanation cache
            batch_size: Number of graphs to process at once
            max_workers: Number of parallel workers (if None, sequential processing)
        """
        self.explainer = explainer
        self.cache = cache
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process_dataset(self,
                       dataset: List[Data],
                       graph_ids: Optional[List[str]] = None,
                       use_cache: bool = True) -> ExplanationBatch:
        """
        Process entire dataset and generate explanations
        
        Args:
            dataset: List of PyG Data objects to explain
            graph_ids: Optional list of graph identifiers
            use_cache: Whether to use cached explanations if available
            
        Returns:
            ExplanationBatch with all explanations
        """
        if graph_ids is None:
            graph_ids = [f"graph_{i}" for i in range(len(dataset))]
        
        if len(graph_ids) != len(dataset):
            raise ValueError("Number of graph IDs must match dataset size")
        
        explanations = []
        cache_hits = 0
        cache_misses = 0
        
        print(f"🔄 Processing {len(dataset)} graphs...")
        start_time = time.time()
        
        for i, (data, graph_id) in enumerate(zip(dataset, graph_ids)):
            explanation = None
            
            # Try cache first
            if use_cache and self.cache is not None:
                explanation = self.cache.load_explanation(
                    graph_id, 
                    self.explainer.model,
                    self.explainer.config
                )
                if explanation is not None:
                    cache_hits += 1
            
            # Generate explanation if not cached
            if explanation is None:
                try:
                    raw_explanation = self.explainer.explain_graph(data)
                    
                    # Process raw explanation (this would use CircuitTracerExplanationProcessor)
                    explanation = self._process_raw_explanation(raw_explanation, data, graph_id)
                    
                    # Cache the result
                    if self.cache is not None:
                        self.cache.save_explanation(
                            explanation,
                            self.explainer.model,
                            self.explainer.config
                        )
                    
                    cache_misses += 1
                    
                except Exception as e:
                    print(f"⚠️ Failed to explain graph {graph_id}: {e}")
                    explanation = None
            
            explanations.append(explanation)
            
            # Progress update
            if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(dataset) - i - 1) / rate if rate > 0 else 0
                
                print(f"  Progress: {i + 1}/{len(dataset)} ({(i + 1)/len(dataset)*100:.1f}%) "
                      f"- {rate:.1f} graphs/sec - ETA: {eta:.0f}s "
                      f"- Cache: {cache_hits} hits, {cache_misses} misses")
        
        total_time = time.time() - start_time
        successful_explanations = sum(1 for exp in explanations if exp is not None)
        
        print(f"✅ Completed in {total_time:.1f}s")
        print(f"   Generated {successful_explanations}/{len(dataset)} explanations")
        print(f"   Cache efficiency: {cache_hits}/{cache_hits + cache_misses} hits")
        
        return ExplanationBatch([exp for exp in explanations if exp is not None])
    
    def _process_raw_explanation(self, 
                                raw_explanation,
                                original_data: Data,
                                graph_id: str) -> AttributionGraphExplanation:
        """
        Convert raw explanation to AttributionGraphExplanation
        
        This is a simplified version - in practice, would use CircuitTracerExplanationProcessor
        """
        # Extract basic information
        if hasattr(raw_explanation, 'prediction'):
            prediction_probs = torch.softmax(raw_explanation.prediction, dim=0)
            predicted_label = raw_explanation.prediction.argmax().item()
            prediction_confidence = prediction_probs.max().item()
        else:
            predicted_label = 1
            prediction_confidence = 0.5
        
        true_label = original_data.y.item() if hasattr(original_data, 'y') else -1
        
        # Extract top edges and nodes
        edge_mask = raw_explanation.edge_mask
        node_mask = raw_explanation.node_mask
        
        # Get top 10% most important edges
        num_important_edges = max(1, int(0.1 * len(edge_mask)))
        top_edge_indices = torch.topk(edge_mask, num_important_edges).indices
        
        important_edges = []
        for idx in top_edge_indices:
            if idx < original_data.edge_index.size(1):
                src, dst = original_data.edge_index[:, idx].tolist()
                importance = edge_mask[idx].item()
                important_edges.append((src, dst, importance))
        
        # Get top 10% most important nodes
        if node_mask.dim() > 1:
            node_importance = node_mask.sum(dim=1)
        else:
            node_importance = node_mask
        
        num_important_nodes = max(1, int(0.1 * len(node_importance)))
        top_node_indices = torch.topk(node_importance, num_important_nodes).indices
        
        important_nodes = []
        for idx in top_node_indices:
            node_idx = idx.item()
            importance = node_importance[idx].item()
            important_nodes.append((node_idx, f"node_{node_idx}", importance))
        
        # Create explanation object
        explanation = AttributionGraphExplanation(
            graph_id=graph_id,
            true_label=true_label,
            predicted_label=predicted_label,
            prediction_confidence=prediction_confidence,
            edge_mask=edge_mask,
            node_mask=node_mask,
            important_edges=important_edges,
            important_nodes=important_nodes,
            critical_features={},
            subgraph_nodes=[node_idx for node_idx, _, _ in important_nodes],
            suspicious_patterns=[],
            circuit_insights={},
            explanation_time=getattr(raw_explanation, 'explanation_time', 0.0),
            sparsity=len(important_edges) / len(edge_mask) if len(edge_mask) > 0 else 0.0
        )
        
        return explanation


def load_model_and_data(model_path: str, 
                       data_path: str,
                       device: str = 'cuda') -> Tuple[Any, List[Data]]:
    """
    Utility function to load trained model and dataset
    
    Args:
        model_path: Path to saved model checkpoint
        data_path: Path to dataset file
        device: Device to load model on
        
    Returns:
        Tuple of (model, dataset)
    """
    print(f"📥 Loading model from {model_path}")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct model (this would need to match your model architecture)
    # For now, return None as placeholder
    model = None  # Would need actual model reconstruction logic
    
    print(f"📥 Loading dataset from {data_path}")
    
    # Load dataset (this would depend on your data format)
    # For now, return empty list as placeholder
    dataset = []  # Would need actual dataset loading logic
    
    return model, dataset


def compute_explanation_statistics(explanations: List[AttributionGraphExplanation]) -> Dict[str, Any]:
    """
    Compute comprehensive statistics across a batch of explanations
    """
    if not explanations:
        return {}
    
    # Basic counts
    total_explanations = len(explanations)
    correct_predictions = sum(1 for exp in explanations if exp.true_label == exp.predicted_label)
    accuracy = correct_predictions / total_explanations
    
    # Label distribution
    benign_count = sum(1 for exp in explanations if exp.true_label == 0)
    injected_count = sum(1 for exp in explanations if exp.true_label == 1)
    
    # Quality metrics
    fidelity_plus_values = [exp.fidelity_plus for exp in explanations]
    fidelity_minus_values = [exp.fidelity_minus for exp in explanations]
    sparsity_values = [exp.sparsity for exp in explanations]
    confidence_values = [exp.prediction_confidence for exp in explanations]
    
    # Pattern analysis
    all_patterns = []
    for exp in explanations:
        all_patterns.extend(exp.suspicious_patterns)
    
    pattern_counts = {}
    for pattern in all_patterns:
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    # Feature importance analysis
    all_features = {}
    for exp in explanations:
        for feature, importance in exp.critical_features.items():
            if feature not in all_features:
                all_features[feature] = []
            all_features[feature].append(importance)
    
    feature_stats = {}
    for feature, values in all_features.items():
        feature_stats[feature] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'count': len(values)
        }
    
    return {
        'summary': {
            'total_explanations': total_explanations,
            'accuracy': accuracy,
            'label_distribution': {
                'benign': benign_count,
                'injected': injected_count
            }
        },
        'quality_metrics': {
            'fidelity_plus': {
                'mean': np.mean(fidelity_plus_values),
                'std': np.std(fidelity_plus_values),
                'median': np.median(fidelity_plus_values)
            },
            'fidelity_minus': {
                'mean': np.mean(fidelity_minus_values),
                'std': np.std(fidelity_minus_values),
                'median': np.median(fidelity_minus_values)
            },
            'sparsity': {
                'mean': np.mean(sparsity_values),
                'std': np.std(sparsity_values),
                'median': np.median(sparsity_values)
            },
            'confidence': {
                'mean': np.mean(confidence_values),
                'std': np.std(confidence_values),
                'median': np.median(confidence_values)
            }
        },
        'patterns': {
            'total_patterns': len(all_patterns),
            'unique_patterns': len(pattern_counts),
            'most_common': sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        },
        'features': feature_stats
    }


def export_explanations_to_csv(explanations: List[AttributionGraphExplanation],
                              output_path: str):
    """
    Export explanation summary to CSV for analysis
    """
    import pandas as pd
    
    rows = []
    for exp in explanations:
        row = {
            'graph_id': exp.graph_id,
            'true_label': exp.true_label,
            'predicted_label': exp.predicted_label,
            'prediction_confidence': exp.prediction_confidence,
            'correct_prediction': exp.true_label == exp.predicted_label,
            'fidelity_plus': exp.fidelity_plus,
            'fidelity_minus': exp.fidelity_minus,
            'sparsity': exp.sparsity,
            'num_important_edges': len(exp.important_edges),
            'num_important_nodes': len(exp.important_nodes),
            'num_critical_features': len(exp.critical_features),
            'num_patterns': len(exp.suspicious_patterns),
            'patterns': '|'.join(exp.suspicious_patterns),
            'explanation_time': exp.explanation_time
        }
        
        # Add top features
        top_features = exp.get_top_features(3)
        for i, (feature, importance) in enumerate(top_features):
            row[f'top_feature_{i+1}'] = feature
            row[f'top_feature_{i+1}_importance'] = importance
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"📊 Exported {len(explanations)} explanations to {output_path}")


def test_utils():
    """Test utility functions"""
    print("Testing explanation utilities...")
    
    try:
        # Test cache
        cache = ExplanationCache("./test_cache")
        stats = cache.get_cache_stats()
        print(f"✅ Cache initialized: {stats}")
        
        # Clean up test cache
        cache.clear_cache()
        
        return True
    except Exception as e:
        print(f"❌ Utils test failed: {e}")
        return False


if __name__ == "__main__":
    test_utils()