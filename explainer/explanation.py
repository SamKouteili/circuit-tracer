"""
Data structures for storing and manipulating attribution graph explanations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import torch
import numpy as np
import json


@dataclass
class AttributionGraphExplanation:
    """
    Comprehensive explanation container for attribution graph predictions
    
    This class stores both raw explanation masks from GNNExplainer and
    processed, domain-specific insights for circuit-tracer interpretability.
    """
    
    # Basic prediction information
    graph_id: str
    true_label: int                      # 0=benign, 1=injected  
    predicted_label: int
    prediction_confidence: float
    
    # Raw explanation masks from GNNExplainer
    edge_mask: torch.Tensor             # [num_edges] - importance of each edge
    node_mask: torch.Tensor             # [num_nodes, num_features] - feature importance per node
    
    # Processed important elements (human-readable)
    important_edges: List[Tuple[int, int, float]] = field(default_factory=list)  # [(src, dst, importance)]
    important_nodes: List[Tuple[int, str, float]] = field(default_factory=list)  # [(node_idx, node_id, importance)]
    critical_features: Dict[str, float] = field(default_factory=dict)            # {feature_name: avg_importance}
    
    # Subgraph extraction results
    explanation_subgraph: Optional[torch.Tensor] = None    # Minimal explaining subgraph edges
    subgraph_nodes: List[int] = field(default_factory=list) # Node indices in explaining subgraph
    
    # Domain-specific circuit-tracer insights  
    suspicious_patterns: List[str] = field(default_factory=list)     # Identified attack patterns
    circuit_insights: Dict[str, Any] = field(default_factory=dict)   # Circuit-specific analysis
    
    # Explanation quality metrics
    fidelity_plus: float = 0.0          # Necessity score (how much pred changes when removing explanation)
    fidelity_minus: float = 0.0         # Sufficiency score (can explanation alone maintain prediction)
    sparsity: float = 0.0               # Compactness of explanation
    
    # Metadata
    explanation_time: float = 0.0       # Time taken to generate explanation
    model_info: Dict[str, Any] = field(default_factory=dict)  # Model config used
    
    def to_dict(self) -> Dict:
        """
        Convert explanation to JSON-serializable dictionary
        
        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            'graph_id': self.graph_id,
            'prediction_info': {
                'true_label': self.true_label,
                'predicted_label': self.predicted_label,
                'prediction_confidence': self.prediction_confidence
            },
            'important_elements': {
                'edges': self.important_edges,
                'nodes': self.important_nodes,
                'features': self.critical_features
            },
            'subgraph': {
                'nodes': self.subgraph_nodes,
                'edge_count': len(self.important_edges)
            },
            'domain_insights': {
                'suspicious_patterns': self.suspicious_patterns,
                'circuit_insights': self.circuit_insights
            },
            'quality_metrics': {
                'fidelity_plus': self.fidelity_plus,
                'fidelity_minus': self.fidelity_minus,
                'sparsity': self.sparsity
            },
            'metadata': {
                'explanation_time': self.explanation_time,
                'model_info': self.model_info
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AttributionGraphExplanation':
        """
        Create explanation from dictionary representation
        
        Args:
            data: Dictionary representation (from to_dict())
            
        Returns:
            AttributionGraphExplanation instance
        """
        return cls(
            graph_id=data['graph_id'],
            true_label=data['prediction_info']['true_label'],
            predicted_label=data['prediction_info']['predicted_label'],
            prediction_confidence=data['prediction_info']['prediction_confidence'],
            edge_mask=torch.empty(0),  # Raw tensors not serialized
            node_mask=torch.empty(0, 0),
            important_edges=data['important_elements']['edges'],
            important_nodes=data['important_elements']['nodes'],
            critical_features=data['important_elements']['features'],
            subgraph_nodes=data['subgraph']['nodes'],
            suspicious_patterns=data['domain_insights']['suspicious_patterns'],
            circuit_insights=data['domain_insights']['circuit_insights'],
            fidelity_plus=data['quality_metrics']['fidelity_plus'],
            fidelity_minus=data['quality_metrics']['fidelity_minus'],
            sparsity=data['quality_metrics']['sparsity'],
            explanation_time=data['metadata']['explanation_time'],
            model_info=data['metadata']['model_info']
        )
    
    def save_json(self, filepath: str):
        """Save explanation to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_json(cls, filepath: str) -> 'AttributionGraphExplanation':
        """Load explanation from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_top_edges(self, k: int = 5) -> List[Tuple[int, int, float]]:
        """Get top-k most important edges"""
        return sorted(self.important_edges, key=lambda x: x[2], reverse=True)[:k]
    
    def get_top_nodes(self, k: int = 5) -> List[Tuple[int, str, float]]:
        """Get top-k most important nodes"""
        return sorted(self.important_nodes, key=lambda x: x[2], reverse=True)[:k]
    
    def get_top_features(self, k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k most important features"""
        return sorted(self.critical_features.items(), key=lambda x: x[1], reverse=True)[:k]
    
    def summary(self) -> str:
        """Generate human-readable summary of explanation"""
        label_names = {0: 'Benign', 1: 'Injected'}
        
        summary = f"""
Attribution Graph Explanation Summary
=====================================

Graph ID: {self.graph_id}
True Label: {label_names.get(self.true_label, 'Unknown')}
Predicted Label: {label_names.get(self.predicted_label, 'Unknown')} 
Confidence: {self.prediction_confidence:.3f}

Key Elements:
- {len(self.important_edges)} important edges
- {len(self.important_nodes)} important nodes  
- {len(self.critical_features)} critical features

Quality Metrics:
- Fidelity+: {self.fidelity_plus:.3f} (necessity)
- Fidelity-: {self.fidelity_minus:.3f} (sufficiency)
- Sparsity: {self.sparsity:.3f} (compactness)

Suspicious Patterns Detected:
{chr(10).join(f"- {pattern}" for pattern in self.suspicious_patterns) if self.suspicious_patterns else "- None detected"}

Top Features:
{chr(10).join(f"- {name}: {importance:.3f}" for name, importance in self.get_top_features(3))}
"""
        return summary
    
    def __repr__(self) -> str:
        return (f"AttributionGraphExplanation(graph_id='{self.graph_id}', "
                f"true_label={self.true_label}, predicted_label={self.predicted_label}, "
                f"confidence={self.prediction_confidence:.3f})")


@dataclass 
class ExplanationBatch:
    """Container for multiple explanations with batch statistics"""
    
    explanations: List[AttributionGraphExplanation]
    
    def __len__(self) -> int:
        return len(self.explanations)
    
    def __getitem__(self, idx: int) -> AttributionGraphExplanation:
        return self.explanations[idx]
    
    def filter_by_label(self, label: int) -> 'ExplanationBatch':
        """Filter explanations by true or predicted label"""
        filtered = [exp for exp in self.explanations 
                   if exp.true_label == label or exp.predicted_label == label]
        return ExplanationBatch(filtered)
    
    def get_accuracy(self) -> float:
        """Compute prediction accuracy across batch"""
        if not self.explanations:
            return 0.0
        correct = sum(1 for exp in self.explanations 
                     if exp.true_label == exp.predicted_label)
        return correct / len(self.explanations)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Compute average quality metrics across batch"""
        if not self.explanations:
            return {}
        
        metrics = {
            'fidelity_plus': np.mean([exp.fidelity_plus for exp in self.explanations]),
            'fidelity_minus': np.mean([exp.fidelity_minus for exp in self.explanations]),
            'sparsity': np.mean([exp.sparsity for exp in self.explanations]),
            'prediction_confidence': np.mean([exp.prediction_confidence for exp in self.explanations]),
            'explanation_time': np.mean([exp.explanation_time for exp in self.explanations])
        }
        
        return metrics
    
    def get_pattern_frequency(self) -> Dict[str, int]:
        """Get frequency of suspicious patterns across batch"""
        pattern_counts = {}
        for exp in self.explanations:
            for pattern in exp.suspicious_patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        return pattern_counts
    
    def save_batch(self, directory: str):
        """Save all explanations in batch to directory"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save individual explanations
        for i, exp in enumerate(self.explanations):
            exp.save_json(os.path.join(directory, f'explanation_{i:04d}.json'))
        
        # Save batch summary
        summary = {
            'batch_size': len(self.explanations),
            'accuracy': self.get_accuracy(),
            'average_metrics': self.get_average_metrics(),
            'pattern_frequency': self.get_pattern_frequency()
        }
        
        with open(os.path.join(directory, 'batch_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)