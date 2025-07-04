"""
Evaluation metrics for explanation quality assessment
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from torch_geometric.data import Data
from sklearn.metrics import jaccard_score
import copy

from .explanation import AttributionGraphExplanation


class ExplanationEvaluator:
    """
    Comprehensive evaluation of graph explanations using multiple quality metrics
    
    Implements standard explainability metrics following GNNExplainer paper (Ying et al., 2019):
    - Fidelity+ (Necessity): How much prediction confidence drops when removing explanation
    - Fidelity- (Sufficiency): How well explanation subgraph alone maintains prediction
    - Sparsity: Compactness of explanation (fraction of graph elements used)
    - Stability: Consistency across similar graphs
    
    All metrics designed for graph-level classification tasks.
    """
    
    def __init__(self, model, device='cuda'):
        """
        Initialize evaluator
        
        Args:
            model: Trained GraphGPS model to evaluate explanations for
            device: Device to run evaluations on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def evaluate_explanation(self, 
                           explanation: AttributionGraphExplanation,
                           original_data: Data,
                           compute_stability: bool = False,
                           similar_graphs: Optional[List[Data]] = None) -> Dict[str, float]:
        """
        Compute all quality metrics for a single explanation
        
        Args:
            explanation: The explanation to evaluate
            original_data: Original graph data that was explained
            compute_stability: Whether to compute stability metric
            similar_graphs: List of similar graphs for stability computation
            
        Returns:
            Dictionary with all computed metrics
        """
        original_data = original_data.to(self.device)
        
        metrics = {}
        
        # Fidelity+ (Necessity): How much does prediction change when removing explanation?
        metrics['fidelity_plus'] = self.compute_fidelity_plus(
            explanation, original_data
        )
        
        # Fidelity- (Sufficiency): Can explanation alone maintain prediction?
        metrics['fidelity_minus'] = self.compute_fidelity_minus(
            explanation, original_data
        )
        
        # Sparsity: How compact is the explanation?
        metrics['sparsity'] = self.compute_sparsity(explanation, original_data)
        
        # Stability: How consistent are explanations for similar graphs?
        if compute_stability and similar_graphs is not None:
            metrics['stability'] = self.compute_stability(
                explanation, original_data, similar_graphs
            )
        
        # Prediction consistency: Does explanation respect model confidence?
        metrics['prediction_consistency'] = self.compute_prediction_consistency(
            explanation, original_data
        )
        
        return metrics
    
    def compute_fidelity_plus(self, 
                             explanation: AttributionGraphExplanation, 
                             original_data: Data) -> float:
        """
        Fidelity+ (Necessity): Measures how much the prediction changes 
        when the explanation elements are removed from the input.
        
        Higher values indicate more necessary explanations.
        """
        with torch.no_grad():
            # Get original prediction
            edge_attr = original_data.edge_attr if hasattr(original_data, 'edge_attr') else None
            original_logits, _ = self.model(
                original_data.x, 
                original_data.edge_index, 
                original_data.batch,
                edge_attr
            )
            original_probs = F.softmax(original_logits, dim=-1)
            
            # Create masked version by removing important elements
            masked_data = self._mask_explanation_elements(original_data, explanation, remove=True)
            
            # Get prediction on masked data
            masked_logits, _ = self.model(
                masked_data.x,
                masked_data.edge_index, 
                masked_data.batch,
                masked_data.edge_attr
            )
            masked_probs = F.softmax(masked_logits, dim=-1)
            
            # Fidelity+ should measure probability DROP when removing explanation
            pred_class = explanation.predicted_label
            fidelity_plus = (original_probs[0, pred_class] - masked_probs[0, pred_class]).item()
            
            # Ensure non-negative (should decrease when removing important elements)
            return max(0.0, fidelity_plus)
    
    def compute_fidelity_minus(self,
                              explanation: AttributionGraphExplanation,
                              original_data: Data) -> float:
        """
        Fidelity- (Sufficiency): Measures how well the explanation subgraph
        alone can maintain the original prediction.
        
        Higher values indicate more sufficient explanations.
        """
        with torch.no_grad():
            # Get original prediction
            edge_attr = original_data.edge_attr if hasattr(original_data, 'edge_attr') else None
            original_logits, _ = self.model(
                original_data.x,
                original_data.edge_index,
                original_data.batch, 
                edge_attr
            )
            original_probs = F.softmax(original_logits, dim=-1)
            
            # Create subgraph with only explanation elements
            subgraph_data = self._extract_explanation_subgraph(original_data, explanation)
            
            if subgraph_data is None or subgraph_data.x.size(0) == 0:
                return 0.0
            
            # Get prediction on subgraph
            try:
                subgraph_logits, _ = self.model(
                    subgraph_data.x,
                    subgraph_data.edge_index,
                    subgraph_data.batch,
                    subgraph_data.edge_attr
                )
                subgraph_probs = F.softmax(subgraph_logits, dim=-1)
                
                # Compute probability similarity for predicted class
                pred_class = explanation.predicted_label
                original_prob = original_probs[0, pred_class].item()
                subgraph_prob = subgraph_probs[0, pred_class].item()
                
                # Fidelity- is how well subgraph maintains original probability
                fidelity_minus = 1.0 - abs(original_prob - subgraph_prob)
                return max(0.0, fidelity_minus)
                
            except Exception:
                # If subgraph cannot be processed (too small, etc.)
                return 0.0
    
    def compute_sparsity(self,
                        explanation: AttributionGraphExplanation,
                        original_data: Data) -> float:
        """
        Sparsity: Fraction of graph elements included in explanation.
        
        Lower values indicate more compact (sparse) explanations.
        """
        total_edges = original_data.edge_index.size(1)
        total_nodes = original_data.x.size(0)
        
        # Count explanation elements
        important_edges = len(explanation.important_edges)
        important_nodes = len(explanation.important_nodes) 
        
        # Weight by typical importance (edges usually more important)
        edge_sparsity = important_edges / max(1, total_edges)
        node_sparsity = important_nodes / max(1, total_nodes)
        
        # Combined sparsity (weighted average)
        sparsity = 0.7 * edge_sparsity + 0.3 * node_sparsity
        
        return sparsity
    
    def compute_stability(self,
                         explanation: AttributionGraphExplanation,
                         original_data: Data,
                         similar_graphs: List[Data],
                         explainer=None) -> float:
        """
        Stability: Jaccard similarity of explanations for similar graphs.
        
        Higher values indicate more stable/consistent explanations.
        """
        if not similar_graphs or explainer is None:
            return 0.0
        
        # Get important edge sets for comparison
        original_edges = set(
            (src, dst) for src, dst, _ in explanation.important_edges
        )
        
        similarities = []
        
        for similar_graph in similar_graphs[:5]:  # Limit to 5 for efficiency
            try:
                # Generate explanation for similar graph
                similar_explanation = explainer.explain_graph(similar_graph)
                
                # Extract important edges
                similar_edges = set(
                    (src, dst) for src, dst, _ in similar_explanation.important_edges  
                )
                
                # Compute Jaccard similarity
                if len(original_edges) == 0 and len(similar_edges) == 0:
                    similarity = 1.0
                elif len(original_edges) == 0 or len(similar_edges) == 0:
                    similarity = 0.0
                else:
                    intersection = len(original_edges & similar_edges)
                    union = len(original_edges | similar_edges)
                    similarity = intersection / union if union > 0 else 0.0
                
                similarities.append(similarity)
                
            except Exception:
                # Skip failed explanations
                continue
        
        return np.mean(similarities) if similarities else 0.0
    
    def compute_prediction_consistency(self,
                                     explanation: AttributionGraphExplanation,
                                     original_data: Data) -> float:
        """
        Prediction Consistency: How well explanation confidence correlates
        with model prediction confidence.
        
        Higher values indicate more consistent explanations.
        """
        model_confidence = explanation.prediction_confidence
        
        # Explanation confidence based on sparsity and importance
        edge_importance_std = 0.0
        if explanation.important_edges:
            importances = [imp for _, _, imp in explanation.important_edges]
            edge_importance_std = np.std(importances)
        
        # High std means focused explanation (high confidence)
        # Low std means diffuse explanation (low confidence)
        explanation_confidence = min(1.0, edge_importance_std * 2.0)
        
        # Consistency is how well these align
        consistency = 1.0 - abs(model_confidence - explanation_confidence)
        return max(0.0, consistency)
    
    def _mask_explanation_elements(self, 
                                  data: Data, 
                                  explanation: AttributionGraphExplanation,
                                  remove: bool = True) -> Data:
        """
        Create version of graph with explanation elements masked/removed
        """
        masked_data = copy.deepcopy(data)
        
        if remove:
            # Remove important edges by setting their weights to zero
            if hasattr(masked_data, 'edge_attr') and masked_data.edge_attr is not None:
                important_edge_indices = set()
                
                # Find edge indices corresponding to important edges
                for src, dst, _ in explanation.important_edges:
                    edge_mask = (masked_data.edge_index[0] == src) & (masked_data.edge_index[1] == dst)
                    indices = torch.where(edge_mask)[0]
                    important_edge_indices.update(indices.tolist())
                
                # Zero out edge weights for important edges
                for idx in important_edge_indices:
                    if idx < masked_data.edge_attr.size(0):
                        masked_data.edge_attr[idx] = 0.0
            
            # Mask important node features by setting to zero
            important_node_indices = [node_idx for node_idx, _, _ in explanation.important_nodes]
            for node_idx in important_node_indices:
                if node_idx < masked_data.x.size(0):
                    masked_data.x[node_idx] = 0.0
        
        return masked_data
    
    def _extract_explanation_subgraph(self,
                                     data: Data,
                                     explanation: AttributionGraphExplanation) -> Optional[Data]:
        """
        Extract subgraph containing only explanation elements
        """
        if not explanation.subgraph_nodes:
            return None
        
        # Get nodes in subgraph
        subgraph_node_indices = torch.tensor(explanation.subgraph_nodes, device=data.x.device)
        
        if len(subgraph_node_indices) == 0:
            return None
        
        # Extract node features
        subgraph_x = data.x[subgraph_node_indices]
        
        # Create node mapping for edge reindexing
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(subgraph_node_indices)}
        
        # Find edges within subgraph
        edge_mask = torch.isin(data.edge_index[0], subgraph_node_indices) & \
                   torch.isin(data.edge_index[1], subgraph_node_indices)
        
        if not edge_mask.any():
            # No edges in subgraph - create minimal graph
            subgraph_edge_index = torch.empty((2, 0), dtype=torch.long, device=data.edge_index.device)
            subgraph_edge_attr = None
        else:
            # Remap edges to new node indices
            old_edges = data.edge_index[:, edge_mask]
            subgraph_edge_index = torch.tensor([
                [node_mapping[src.item()], node_mapping[dst.item()]]
                for src, dst in old_edges.t()
            ], device=data.edge_index.device).t()
            
            # Extract corresponding edge attributes
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                subgraph_edge_attr = data.edge_attr[edge_mask]
            else:
                subgraph_edge_attr = None
        
        # Create subgraph data object
        subgraph_data = Data(
            x=subgraph_x,
            edge_index=subgraph_edge_index,
            edge_attr=subgraph_edge_attr,
            batch=torch.zeros(subgraph_x.size(0), dtype=torch.long, device=subgraph_x.device)
        )
        
        return subgraph_data
    
    def evaluate_batch(self,
                      explanations: List[AttributionGraphExplanation],
                      original_data_list: List[Data]) -> Dict[str, Any]:
        """
        Evaluate a batch of explanations and compute aggregate statistics
        """
        if len(explanations) != len(original_data_list):
            raise ValueError("Number of explanations must match number of data objects")
        
        all_metrics = []
        
        for explanation, data in zip(explanations, original_data_list):
            if explanation is not None:
                metrics = self.evaluate_explanation(explanation, data)
                all_metrics.append(metrics)
        
        if not all_metrics:
            return {}
        
        # Compute aggregate statistics
        aggregate_stats = {}
        
        for metric_name in all_metrics[0].keys():
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            aggregate_stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        aggregate_stats['num_explanations'] = len(all_metrics)
        aggregate_stats['success_rate'] = len(all_metrics) / len(explanations)
        
        return aggregate_stats
    
    def compare_explanations(self,
                           explanation1: AttributionGraphExplanation,
                           explanation2: AttributionGraphExplanation) -> Dict[str, float]:
        """
        Compare two explanations for similarity
        """
        # Edge overlap
        edges1 = set((src, dst) for src, dst, _ in explanation1.important_edges)
        edges2 = set((src, dst) for src, dst, _ in explanation2.important_edges)
        
        edge_jaccard = len(edges1 & edges2) / len(edges1 | edges2) if edges1 or edges2 else 1.0
        
        # Node overlap
        nodes1 = set(node_idx for node_idx, _, _ in explanation1.important_nodes)
        nodes2 = set(node_idx for node_idx, _, _ in explanation2.important_nodes)
        
        node_jaccard = len(nodes1 & nodes2) / len(nodes1 | nodes2) if nodes1 or nodes2 else 1.0
        
        # Feature overlap
        features1 = set(explanation1.critical_features.keys())
        features2 = set(explanation2.critical_features.keys())
        
        feature_jaccard = len(features1 & features2) / len(features1 | features2) if features1 or features2 else 1.0
        
        # Pattern overlap
        patterns1 = set(explanation1.suspicious_patterns)
        patterns2 = set(explanation2.suspicious_patterns)
        
        pattern_jaccard = len(patterns1 & patterns2) / len(patterns1 | patterns2) if patterns1 or patterns2 else 1.0
        
        return {
            'edge_similarity': edge_jaccard,
            'node_similarity': node_jaccard,
            'feature_similarity': feature_jaccard,
            'pattern_similarity': pattern_jaccard,
            'overall_similarity': np.mean([edge_jaccard, node_jaccard, feature_jaccard, pattern_jaccard])
        }


def test_evaluator():
    """Test the explanation evaluator with dummy data"""
    print("Testing ExplanationEvaluator...")
    
    # Create dummy model and data (simplified for testing)
    import torch.nn as nn
    from torch_geometric.nn import global_mean_pool
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(9, 32)
            self.classifier = nn.Linear(32, 2)
        
        def forward(self, x, edge_index, batch, edge_attr=None):
            h = self.proj(x)
            h_graph = global_mean_pool(h, batch)
            logits = self.classifier(h_graph)
            return logits, h
    
    # Create dummy explanation
    from .explanation import AttributionGraphExplanation
    
    explanation = AttributionGraphExplanation(
        graph_id="test_graph",
        true_label=1,
        predicted_label=1,
        prediction_confidence=0.85,
        edge_mask=torch.randn(100),
        node_mask=torch.randn(50, 9),
        important_edges=[(0, 1, 0.9), (2, 3, 0.8), (4, 5, 0.7)],
        important_nodes=[(0, "node_0", 0.9), (1, "node_1", 0.8)],
        critical_features={"influence": 0.8, "activation": 0.6},
        subgraph_nodes=[0, 1, 2, 3, 4, 5]
    )
    
    # Create dummy data
    num_nodes = 50
    x = torch.randn(num_nodes, 9)
    edge_index = torch.randint(0, num_nodes, (2, 100))
    edge_attr = torch.randn(100, 1)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    # Test evaluator
    model = DummyModel()
    evaluator = ExplanationEvaluator(model, device='cpu')
    
    try:
        metrics = evaluator.evaluate_explanation(explanation, data)
        print("✅ Evaluation completed successfully")
        print("Metrics computed:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")
        return True
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False


if __name__ == "__main__":
    test_evaluator()