"""
Domain-specific processing for converting raw GNNExplainer output 
to circuit-tracer attribution graph insights
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from explanation import AttributionGraphExplanation


class CircuitTracerExplanationProcessor:
    """
    Convert raw GNNExplainer masks to domain-specific circuit-tracer insights
    
    This processor understands the structure and semantics of attribution graphs
    and can identify patterns specific to prompt injection attacks.
    """
    
    def __init__(self, converter=None, vocab_mapping: Optional[Dict[int, str]] = None):
        """
        Initialize the processor
        
        Args:
            converter: AttributionGraphConverter instance (for feature names)
            vocab_mapping: Mapping from node indices to human-readable node IDs
        """
        self.converter = converter
        self.vocab_mapping = vocab_mapping or {}
        
        # Circuit-tracer feature names (from data_converter.py)
        self.feature_names = [
            'influence', 'activation', 'layer', 'ctx_idx', 'feature',
            'is_cross_layer_transcoder', 'is_mlp_error', 'is_embedding', 'is_target_logit'
        ]
        
        # Pattern detection thresholds (tunable)
        self.pattern_thresholds = {
            'high_influence': 0.5,
            'context_manipulation': 0.3,
            'cross_layer_attack': 0.4,
            'logit_manipulation': 0.2,
            'activation_anomaly': 0.4
        }
    
    def process_explanation(self, 
                          raw_explanation: Any, 
                          original_data: torch.Tensor,
                          graph_id: Optional[str] = None) -> AttributionGraphExplanation:
        """
        Convert raw GNNExplainer output to AttributionGraphExplanation
        
        Args:
            raw_explanation: PyG Explanation object from GNNExplainer
            original_data: Original PyG Data object that was explained
            graph_id: Optional identifier for the graph
            
        Returns:
            AttributionGraphExplanation with processed insights
        """
        
        # Extract basic prediction information
        if hasattr(raw_explanation, 'prediction'):
            prediction_probs = torch.softmax(raw_explanation.prediction, dim=0)
            predicted_label = raw_explanation.prediction.argmax().item()
            prediction_confidence = prediction_probs.max().item()
        else:
            # Fallback: get prediction from model directly
            predicted_label = 1  # Default
            prediction_confidence = 0.5
        
        true_label = original_data.y.item() if hasattr(original_data, 'y') else -1
        
        # Process edge importance
        edge_importance = raw_explanation.edge_mask
        important_edges = self._extract_important_edges(edge_importance, original_data.edge_index)
        
        # Process node importance 
        node_importance = self._aggregate_node_importance(raw_explanation.node_mask)
        important_nodes = self._extract_important_nodes(node_importance)
        
        # Analyze feature importance
        critical_features = self._analyze_feature_importance(raw_explanation.node_mask)
        
        # Extract explaining subgraph
        subgraph_nodes = self._extract_subgraph_nodes(edge_importance, original_data.edge_index)
        
        # Identify suspicious patterns
        suspicious_patterns = self._identify_attack_patterns(critical_features, important_edges, important_nodes)
        
        # Extract circuit-specific insights
        circuit_insights = self._extract_circuit_insights(
            important_edges, important_nodes, critical_features, original_data
        )
        
        # Create explanation object
        explanation = AttributionGraphExplanation(
            graph_id=graph_id or f"graph_{id(original_data)}",
            true_label=true_label,
            predicted_label=predicted_label,
            prediction_confidence=prediction_confidence,
            edge_mask=edge_importance,
            node_mask=raw_explanation.node_mask,
            important_edges=important_edges,
            important_nodes=important_nodes,
            critical_features=critical_features,
            subgraph_nodes=subgraph_nodes,
            suspicious_patterns=suspicious_patterns,
            circuit_insights=circuit_insights,
            explanation_time=getattr(raw_explanation, 'explanation_time', 0.0),
            sparsity=len(important_edges) / len(edge_importance) if len(edge_importance) > 0 else 0.0
        )
        
        return explanation
    
    def _extract_important_edges(self, 
                                edge_mask: torch.Tensor, 
                                edge_index: torch.Tensor,
                                top_k_percent: float = 0.1) -> List[Tuple[int, int, float]]:
        """Extract top-k% most important edges"""
        
        if len(edge_mask) == 0:
            return []
        
        # Get top edges by importance
        num_important = max(1, int(top_k_percent * len(edge_mask)))
        top_indices = torch.topk(edge_mask, num_important).indices
        
        important_edges = []
        for idx in top_indices:
            src, dst = edge_index[:, idx].tolist()
            importance = edge_mask[idx].item()
            important_edges.append((src, dst, importance))
        
        return important_edges
    
    def _aggregate_node_importance(self, node_mask: torch.Tensor) -> torch.Tensor:
        """Aggregate node feature importance into per-node scores"""
        if node_mask.dim() == 1:
            return node_mask
        else:
            # Sum importance across all features for each node
            return node_mask.sum(dim=1)
    
    def _extract_important_nodes(self, 
                                node_importance: torch.Tensor,
                                top_k_percent: float = 0.1) -> List[Tuple[int, str, float]]:
        """Extract top-k% most important nodes"""
        
        if len(node_importance) == 0:
            return []
        
        # Get top nodes by importance
        num_important = max(1, int(top_k_percent * len(node_importance)))
        top_indices = torch.topk(node_importance, num_important).indices
        
        important_nodes = []
        for idx in top_indices:
            node_idx = idx.item()
            node_id = self.vocab_mapping.get(node_idx, f"node_{node_idx}")
            importance = node_importance[idx].item()
            important_nodes.append((node_idx, node_id, importance))
        
        return important_nodes
    
    def _analyze_feature_importance(self, node_mask: torch.Tensor) -> Dict[str, float]:
        """Analyze importance of each feature type across all nodes"""
        
        if node_mask.dim() == 1 or node_mask.size(1) != len(self.feature_names):
            # If not the expected shape, return empty
            return {}
        
        # Average importance across all nodes for each feature
        feature_importance = node_mask.mean(dim=0)
        
        critical_features = {}
        for i, feature_name in enumerate(self.feature_names):
            importance = feature_importance[i].item()
            if importance > 0.1:  # Threshold for significance
                critical_features[feature_name] = importance
        
        return critical_features
    
    def _extract_subgraph_nodes(self, 
                               edge_mask: torch.Tensor,
                               edge_index: torch.Tensor,
                               top_k_percent: float = 0.1) -> List[int]:
        """Extract nodes involved in the most important edges"""
        
        if len(edge_mask) == 0:
            return []
        
        # Get top edges
        num_important = max(1, int(top_k_percent * len(edge_mask)))
        top_edge_indices = torch.topk(edge_mask, num_important).indices
        
        # Extract unique nodes from these edges
        important_edge_index = edge_index[:, top_edge_indices]
        unique_nodes = torch.unique(important_edge_index).tolist()
        
        return unique_nodes
    
    def _identify_attack_patterns(self, 
                                 features: Dict[str, float],
                                 edges: List[Tuple[int, int, float]],
                                 nodes: List[Tuple[int, str, float]]) -> List[str]:
        """
        Identify known prompt injection attack patterns based on explanation
        
        This method looks for specific combinations of feature importance
        that are characteristic of different types of attacks.
        """
        patterns = []
        
        # Pattern 1: High influence concentration (attention hijacking)
        if features.get('influence', 0) > self.pattern_thresholds['high_influence']:
            patterns.append('high_influence_concentration')
        
        # Pattern 2: Context position manipulation
        if features.get('ctx_idx', 0) > self.pattern_thresholds['context_manipulation']:
            patterns.append('context_position_manipulation')
        
        # Pattern 3: Cross-layer attack (unusual layer patterns)
        if features.get('layer', 0) > self.pattern_thresholds['cross_layer_attack']:
            patterns.append('cross_layer_attack')
        
        # Pattern 4: Direct logit manipulation
        if features.get('is_target_logit', 0) > self.pattern_thresholds['logit_manipulation']:
            patterns.append('direct_logit_manipulation')
        
        # Pattern 5: Activation anomalies
        if features.get('activation', 0) > self.pattern_thresholds['activation_anomaly']:
            patterns.append('activation_anomaly')
        
        # Pattern 6: Cross-layer transcoder involvement
        if features.get('is_cross_layer_transcoder', 0) > 0.3:
            patterns.append('transcoder_manipulation')
        
        # Pattern 7: High edge density (potential bypass attempt)
        if len(edges) > 20:  # Threshold for "many important edges"
            patterns.append('high_connectivity_pattern')
        
        # Pattern 8: Feature interaction anomaly (multiple critical features)
        if len([f for f in features.values() if f > 0.3]) >= 3:
            patterns.append('multi_feature_attack')
        
        return patterns
    
    def _extract_circuit_insights(self, 
                                 edges: List[Tuple[int, int, float]],
                                 nodes: List[Tuple[int, str, float]],
                                 features: Dict[str, float],
                                 original_data: torch.Tensor) -> Dict[str, Any]:
        """
        Extract circuit-tracer specific insights from the explanation
        
        This method provides deeper analysis that maps to circuit analysis concepts.
        """
        insights = {
            'attack_depth': 0,
            'affected_layers': set(),
            'manipulation_type': 'unknown',
            'confidence': 0.0,
            'key_pathways': [],
            'anomaly_score': 0.0
        }
        
        # Analyze attack depth (how many layers are involved)
        layer_importance = features.get('layer', 0)
        if layer_importance > 0.5:
            insights['attack_depth'] = 'deep'
        elif layer_importance > 0.2:
            insights['attack_depth'] = 'medium'
        else:
            insights['attack_depth'] = 'shallow'
        
        # Determine manipulation type based on feature patterns
        if features.get('is_target_logit', 0) > 0.3:
            insights['manipulation_type'] = 'direct_logit'
        elif features.get('ctx_idx', 0) > 0.4:
            insights['manipulation_type'] = 'context_hijacking'
        elif features.get('influence', 0) > 0.5:
            insights['manipulation_type'] = 'attention_steering'
        elif features.get('activation', 0) > 0.4:
            insights['manipulation_type'] = 'activation_patching'
        
        # Calculate confidence based on explanation quality
        feature_count = len([f for f in features.values() if f > 0.2])
        edge_count = len(edges)
        insights['confidence'] = min(1.0, (feature_count * 0.2 + edge_count * 0.01))
        
        # Identify key attack pathways (most important edge sequences)
        if len(edges) >= 2:
            insights['key_pathways'] = self._find_attack_pathways(edges)
        
        # Calculate anomaly score (how unusual this pattern is)
        insights['anomaly_score'] = self._calculate_anomaly_score(features, edges, nodes)
        
        # Convert set to list for JSON serialization
        insights['affected_layers'] = list(insights['affected_layers'])
        
        return insights
    
    def _find_attack_pathways(self, edges: List[Tuple[int, int, float]]) -> List[List[int]]:
        """Find connected pathways in the important edges"""
        # Simple pathway detection - find chains of connected edges
        pathways = []
        
        # Build adjacency list
        graph = {}
        for src, dst, importance in edges:
            if src not in graph:
                graph[src] = []
            graph[src].append((dst, importance))
        
        # Find paths (simplified - could be more sophisticated)
        visited = set()
        for src in graph:
            if src not in visited:
                path = self._dfs_pathway(graph, src, visited, max_length=5)
                if len(path) > 1:
                    pathways.append(path)
        
        return pathways[:3]  # Return top 3 pathways
    
    def _dfs_pathway(self, graph: Dict, node: int, visited: Set[int], max_length: int = 5) -> List[int]:
        """Simple DFS to find pathways"""
        if node in visited or max_length <= 0:
            return []
        
        visited.add(node)
        path = [node]
        
        if node in graph:
            # Follow the edge with highest importance
            best_neighbor = max(graph[node], key=lambda x: x[1])[0]
            path.extend(self._dfs_pathway(graph, best_neighbor, visited, max_length - 1))
        
        return path
    
    def _calculate_anomaly_score(self, 
                                features: Dict[str, float],
                                edges: List[Tuple[int, int, float]],
                                nodes: List[Tuple[int, str, float]]) -> float:
        """Calculate how anomalous/unusual this explanation pattern is"""
        
        score = 0.0
        
        # High feature diversity suggests complex attack
        active_features = len([f for f in features.values() if f > 0.2])
        score += min(1.0, active_features / 5.0) * 0.3
        
        # High edge importance concentration suggests focused attack
        if edges:
            edge_importance_std = np.std([importance for _, _, importance in edges])
            score += min(1.0, edge_importance_std * 2) * 0.3
        
        # Cross-layer transcoder involvement is unusual
        if features.get('is_cross_layer_transcoder', 0) > 0.3:
            score += 0.2
        
        # Multiple attack patterns suggest sophisticated attack
        pattern_diversity = len(self._identify_attack_patterns(features, edges, nodes))
        score += min(1.0, pattern_diversity / 4.0) * 0.2
        
        return min(1.0, score)
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update pattern detection thresholds"""
        self.pattern_thresholds.update(new_thresholds)
    
    def get_pattern_statistics(self, explanations: List[AttributionGraphExplanation]) -> Dict[str, Any]:
        """Analyze patterns across multiple explanations"""
        
        all_patterns = []
        feature_importance_stats = {}
        
        for exp in explanations:
            all_patterns.extend(exp.suspicious_patterns)
            
            for feature, importance in exp.critical_features.items():
                if feature not in feature_importance_stats:
                    feature_importance_stats[feature] = []
                feature_importance_stats[feature].append(importance)
        
        # Calculate pattern frequencies
        pattern_counts = {}
        for pattern in all_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Calculate feature importance statistics
        feature_stats = {}
        for feature, values in feature_importance_stats.items():
            feature_stats[feature] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'max': np.max(values),
                'count': len(values)
            }
        
        return {
            'pattern_frequencies': pattern_counts,
            'feature_statistics': feature_stats,
            'total_explanations': len(explanations)
        }