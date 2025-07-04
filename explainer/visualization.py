"""
Visualization tools for attribution graph explanations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import os

from .explanation import AttributionGraphExplanation


class ExplanationVisualizer:
    """
    Comprehensive visualization tools for graph explanations
    
    Provides multiple visualization types:
    - Subgraph importance plots
    - Feature importance bar charts
    - Pattern detection summaries
    - Comparative analysis plots
    """
    
    def __init__(self, 
                 feature_names: Optional[List[str]] = None,
                 node_vocab: Optional[Dict[int, str]] = None):
        """
        Initialize visualizer
        
        Args:
            feature_names: Names of node features for labeling
            node_vocab: Mapping from node indices to human-readable names
        """
        self.feature_names = feature_names or [
            'influence', 'activation', 'layer', 'ctx_idx', 'feature',
            'is_cross_layer_transcoder', 'is_mlp_error', 'is_embedding', 'is_target_logit'
        ]
        self.node_vocab = node_vocab or {}
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Custom colormap for importance visualization
        self.importance_cmap = LinearSegmentedColormap.from_list(
            'importance', ['lightblue', 'yellow', 'orange', 'red']
        )
    
    def plot_subgraph_explanation(self, 
                                 explanation: AttributionGraphExplanation,
                                 original_data: torch.Tensor,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 8),
                                 top_k_edges: int = 20,
                                 top_k_nodes: int = 15) -> plt.Figure:
        """
        Visualize the important subgraph with edge and node importance
        
        Args:
            explanation: Explanation object to visualize
            original_data: Original PyG Data object
            save_path: Path to save the plot
            figsize: Figure size (width, height)
            top_k_edges: Number of top edges to highlight
            top_k_nodes: Number of top nodes to highlight
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Create NetworkX graph for visualization
        G = nx.Graph()
        
        # Add all nodes from subgraph
        subgraph_nodes = explanation.subgraph_nodes[:top_k_nodes] if explanation.subgraph_nodes else []
        for node_idx in subgraph_nodes:
            node_name = self.node_vocab.get(node_idx, f"n{node_idx}")
            G.add_node(node_idx, name=node_name)
        
        # Add important edges
        edge_weights = {}
        for src, dst, importance in explanation.get_top_edges(top_k_edges):
            if src in subgraph_nodes and dst in subgraph_nodes:
                G.add_edge(src, dst)
                edge_weights[(src, dst)] = importance
        
        if len(G.nodes()) == 0:
            ax1.text(0.5, 0.5, 'No subgraph nodes to display', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title(f"Subgraph Explanation: {explanation.graph_id}")
            ax2.axis('off')
            return fig
        
        # Layout the graph
        try:
            pos = nx.spring_layout(G, k=1, iterations=50)
        except:
            pos = nx.random_layout(G)
        
        # Plot 1: Subgraph with edge importance
        ax1.set_title(f"Important Subgraph\n{explanation.graph_id}")
        
        # Draw nodes with importance-based sizing
        node_importances = {node_idx: imp for node_idx, _, imp in explanation.get_top_nodes(top_k_nodes)}
        
        for node in G.nodes():
            importance = node_importances.get(node, 0.0)
            node_size = 300 + importance * 1000  # Scale size by importance
            node_color = self.importance_cmap(importance)
            
            ax1.scatter(pos[node][0], pos[node][1], 
                       s=node_size, c=[node_color], 
                       alpha=0.8, edgecolors='black', linewidth=1)
            
            # Add node labels
            ax1.annotate(f"{node}", pos[node], xytext=(5, 5), 
                        textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        # Draw edges with importance-based width and color
        for edge in G.edges():
            src, dst = edge
            importance = edge_weights.get(edge, edge_weights.get((dst, src), 0.0))
            
            edge_width = 1 + importance * 5  # Scale width by importance
            edge_color = self.importance_cmap(importance)
            
            ax1.plot([pos[src][0], pos[dst][0]], 
                    [pos[src][1], pos[dst][1]],
                    color=edge_color, linewidth=edge_width, alpha=0.7)
        
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        ax1.axis('off')
        
        # Add importance colorbar
        sm = plt.cm.ScalarMappable(cmap=self.importance_cmap, 
                                  norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, shrink=0.6)
        cbar.set_label('Importance', rotation=270, labelpad=15)
        
        # Plot 2: Prediction info and metrics
        ax2.axis('off')
        
        # Prediction information
        labels = {0: 'Benign', 1: 'Injected'}
        true_label_name = labels.get(explanation.true_label, 'Unknown')
        pred_label_name = labels.get(explanation.predicted_label, 'Unknown')
        
        info_text = f"""
Prediction Information:
• True Label: {true_label_name}
• Predicted: {pred_label_name}
• Confidence: {explanation.prediction_confidence:.3f}

Explanation Quality:
• Fidelity+: {explanation.fidelity_plus:.3f}
• Fidelity-: {explanation.fidelity_minus:.3f}
• Sparsity: {explanation.sparsity:.3f}

Key Elements:
• {len(explanation.important_edges)} important edges
• {len(explanation.important_nodes)} important nodes
• {len(explanation.critical_features)} critical features

Suspicious Patterns:
{chr(10).join(f"• {pattern}" for pattern in explanation.suspicious_patterns[:5]) if explanation.suspicious_patterns else "• None detected"}
        """
        
        ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Subgraph explanation saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(self,
                              explanation: AttributionGraphExplanation,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Create bar plot of feature importance across the graph
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if not explanation.critical_features:
            ax.text(0.5, 0.5, 'No critical features detected', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Feature Importance Analysis")
            return fig
        
        # Sort features by importance
        sorted_features = sorted(explanation.critical_features.items(), 
                               key=lambda x: x[1], reverse=True)
        
        features, importances = zip(*sorted_features)
        
        # Create color gradient
        colors = [self.importance_cmap(imp) for imp in importances]
        
        # Create bar plot
        bars = ax.barh(range(len(features)), importances, color=colors, alpha=0.8)
        
        # Customize plot
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([f.replace('_', ' ').title() for f in features])
        ax.set_xlabel('Average Importance')
        ax.set_title(f'Feature Importance Analysis\n{explanation.graph_id}')
        
        # Add value labels on bars
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{imp:.3f}', va='center', fontsize=9)
        
        # Add threshold line for significance
        ax.axvline(x=0.1, color='red', linestyle='--', alpha=0.7, 
                  label='Significance threshold')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        return fig
    
    def plot_pattern_analysis(self,
                            explanations: List[AttributionGraphExplanation],
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Analyze patterns across multiple explanations
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Pattern frequency across explanations
        all_patterns = []
        for exp in explanations:
            all_patterns.extend(exp.suspicious_patterns)
        
        if all_patterns:
            pattern_counts = {}
            for pattern in all_patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
            patterns, counts = zip(*sorted_patterns[:10])  # Top 10 patterns
            
            ax1.barh(range(len(patterns)), counts, alpha=0.7)
            ax1.set_yticks(range(len(patterns)))
            ax1.set_yticklabels([p.replace('_', ' ').title() for p in patterns])
            ax1.set_xlabel('Frequency')
            ax1.set_title('Most Common Suspicious Patterns')
        else:
            ax1.text(0.5, 0.5, 'No patterns detected', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Suspicious Patterns')
        
        # 2. Prediction accuracy by pattern
        pattern_accuracy = {}
        for exp in explanations:
            is_correct = (exp.true_label == exp.predicted_label)
            for pattern in exp.suspicious_patterns:
                if pattern not in pattern_accuracy:
                    pattern_accuracy[pattern] = []
                pattern_accuracy[pattern].append(is_correct)
        
        if pattern_accuracy:
            pattern_acc_means = {p: np.mean(acc) for p, acc in pattern_accuracy.items()}
            sorted_acc = sorted(pattern_acc_means.items(), key=lambda x: x[1], reverse=True)
            
            if sorted_acc:
                patterns_acc, accs = zip(*sorted_acc[:8])
                ax2.bar(range(len(patterns_acc)), accs, alpha=0.7)
                ax2.set_xticks(range(len(patterns_acc)))
                ax2.set_xticklabels([p.replace('_', ' ')[:10] for p in patterns_acc], rotation=45)
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Prediction Accuracy by Pattern')
                ax2.set_ylim(0, 1)
        else:
            ax2.text(0.5, 0.5, 'No pattern accuracy data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Pattern Accuracy')
        
        # 3. Feature importance distribution
        all_features = {}
        for exp in explanations:
            for feature, importance in exp.critical_features.items():
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(importance)
        
        if all_features:
            feature_means = {f: np.mean(vals) for f, vals in all_features.items()}
            sorted_features = sorted(feature_means.items(), key=lambda x: x[1], reverse=True)
            
            features, means = zip(*sorted_features[:8])
            stds = [np.std(all_features[f]) for f in features]
            
            ax3.bar(range(len(features)), means, yerr=stds, alpha=0.7, capsize=5)
            ax3.set_xticks(range(len(features)))
            ax3.set_xticklabels([f.replace('_', ' ')[:10] for f in features], rotation=45)
            ax3.set_ylabel('Mean Importance')
            ax3.set_title('Feature Importance Distribution')
        else:
            ax3.text(0.5, 0.5, 'No feature data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Feature Importance')
        
        # 4. Explanation quality metrics
        metrics = ['fidelity_plus', 'fidelity_minus', 'sparsity', 'prediction_confidence']
        metric_values = {metric: [] for metric in metrics}
        
        for exp in explanations:
            metric_values['fidelity_plus'].append(exp.fidelity_plus)
            metric_values['fidelity_minus'].append(exp.fidelity_minus)
            metric_values['sparsity'].append(exp.sparsity)
            metric_values['prediction_confidence'].append(exp.prediction_confidence)
        
        positions = range(len(metrics))
        means = [np.mean(metric_values[m]) for m in metrics]
        stds = [np.std(metric_values[m]) for m in metrics]
        
        ax4.bar(positions, means, yerr=stds, alpha=0.7, capsize=5)
        ax4.set_xticks(positions)
        ax4.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
        ax4.set_ylabel('Value')
        ax4.set_title('Explanation Quality Metrics')
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Pattern analysis saved to {save_path}")
        
        return fig
    
    def create_explanation_report(self,
                                explanation: AttributionGraphExplanation,
                                original_data: torch.Tensor,
                                save_path: Optional[str] = None,
                                include_subgraph: bool = True,
                                include_features: bool = True) -> str:
        """
        Generate comprehensive HTML report for explanation
        """
        if save_path is None:
            save_path = f"explanation_report_{explanation.graph_id}.html"
        
        # Generate individual plots
        base_name = os.path.splitext(save_path)[0]
        
        subgraph_path = None
        feature_path = None
        
        if include_subgraph:
            subgraph_path = f"{base_name}_subgraph.png"
            self.plot_subgraph_explanation(explanation, original_data, subgraph_path)
        
        if include_features:
            feature_path = f"{base_name}_features.png"
            self.plot_feature_importance(explanation, feature_path)
        
        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Attribution Graph Explanation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 5px; }}
        .pattern {{ background-color: #fff3cd; padding: 5px; margin: 5px 0; border-radius: 3px; }}
        .image {{ text-align: center; margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Attribution Graph Explanation Report</h1>
        <h2>Graph ID: {explanation.graph_id}</h2>
    </div>
    
    <div class="section">
        <h3>Prediction Summary</h3>
        <div class="metric">
            <strong>True Label:</strong> {'Benign' if explanation.true_label == 0 else 'Injected'}
        </div>
        <div class="metric">
            <strong>Predicted Label:</strong> {'Benign' if explanation.predicted_label == 0 else 'Injected'}
        </div>
        <div class="metric">
            <strong>Confidence:</strong> {explanation.prediction_confidence:.3f}
        </div>
        <div class="metric">
            <strong>Correct:</strong> {'✓' if explanation.true_label == explanation.predicted_label else '✗'}
        </div>
    </div>
    
    <div class="section">
        <h3>Explanation Quality Metrics</h3>
        <div class="metric">
            <strong>Fidelity+ (Necessity):</strong> {explanation.fidelity_plus:.3f}
        </div>
        <div class="metric">
            <strong>Fidelity- (Sufficiency):</strong> {explanation.fidelity_minus:.3f}
        </div>
        <div class="metric">
            <strong>Sparsity:</strong> {explanation.sparsity:.3f}
        </div>
        <div class="metric">
            <strong>Generation Time:</strong> {explanation.explanation_time:.2f}s
        </div>
    </div>
    
    <div class="section">
        <h3>Suspicious Patterns Detected</h3>
        {chr(10).join(f'<div class="pattern">• {pattern.replace("_", " ").title()}</div>' 
                     for pattern in explanation.suspicious_patterns) 
         if explanation.suspicious_patterns else '<p>No suspicious patterns detected.</p>'}
    </div>
    
    <div class="section">
        <h3>Important Elements</h3>
        <table>
            <tr><th>Element Type</th><th>Count</th><th>Top Examples</th></tr>
            <tr>
                <td>Edges</td>
                <td>{len(explanation.important_edges)}</td>
                <td>{', '.join(f'({src}→{dst}: {imp:.3f})' for src, dst, imp in explanation.get_top_edges(3))}</td>
            </tr>
            <tr>
                <td>Nodes</td>
                <td>{len(explanation.important_nodes)}</td>
                <td>{', '.join(f'{node_id} ({imp:.3f})' for _, node_id, imp in explanation.get_top_nodes(3))}</td>
            </tr>
            <tr>
                <td>Features</td>
                <td>{len(explanation.critical_features)}</td>
                <td>{', '.join(f'{feat} ({imp:.3f})' for feat, imp in explanation.get_top_features(3))}</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h3>Circuit Insights</h3>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            {chr(10).join(f'<tr><td>{key.replace("_", " ").title()}</td><td>{value}</td></tr>' 
                         for key, value in explanation.circuit_insights.items())}
        </table>
    </div>
"""
        
        if subgraph_path and os.path.exists(subgraph_path):
            html_content += f"""
    <div class="section">
        <h3>Subgraph Visualization</h3>
        <div class="image">
            <img src="{os.path.basename(subgraph_path)}" alt="Subgraph Explanation" style="max-width: 100%; height: auto;">
        </div>
    </div>
"""
        
        if feature_path and os.path.exists(feature_path):
            html_content += f"""
    <div class="section">
        <h3>Feature Importance Analysis</h3>
        <div class="image">
            <img src="{os.path.basename(feature_path)}" alt="Feature Importance" style="max-width: 100%; height: auto;">
        </div>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        # Save HTML report
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"Comprehensive explanation report saved to {save_path}")
        return save_path
    
    def plot_explanation_comparison(self,
                                  explanations: List[AttributionGraphExplanation],
                                  labels: Optional[List[str]] = None,
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Compare multiple explanations side by side
        """
        n_explanations = len(explanations)
        if n_explanations == 0:
            raise ValueError("No explanations provided")
        
        fig, axes = plt.subplots(2, n_explanations, figsize=figsize)
        if n_explanations == 1:
            axes = axes.reshape(-1, 1)
        
        labels = labels or [f"Exp {i+1}" for i in range(n_explanations)]
        
        for i, (exp, label) in enumerate(zip(explanations, labels)):
            # Top row: Feature importance comparison
            ax_feat = axes[0, i]
            
            if exp.critical_features:
                features, importances = zip(*sorted(exp.critical_features.items(), 
                                                  key=lambda x: x[1], reverse=True)[:5])
                colors = [self.importance_cmap(imp) for imp in importances]
                ax_feat.barh(range(len(features)), importances, color=colors, alpha=0.8)
                ax_feat.set_yticks(range(len(features)))
                ax_feat.set_yticklabels([f.replace('_', ' ')[:10] for f in features], fontsize=8)
                ax_feat.set_xlabel('Importance')
            else:
                ax_feat.text(0.5, 0.5, 'No features', ha='center', va='center', transform=ax_feat.transAxes)
            
            ax_feat.set_title(f'{label}\nFeature Importance')
            
            # Bottom row: Metrics comparison
            ax_metrics = axes[1, i]
            
            metrics = ['Fidelity+', 'Fidelity-', 'Sparsity', 'Confidence']
            values = [exp.fidelity_plus, exp.fidelity_minus, exp.sparsity, exp.prediction_confidence]
            
            bars = ax_metrics.bar(metrics, values, alpha=0.7)
            ax_metrics.set_ylim(0, 1)
            ax_metrics.set_title('Quality Metrics')
            ax_metrics.tick_params(axis='x', rotation=45)
            
            # Color bars by value
            for bar, value in zip(bars, values):
                bar.set_color(self.importance_cmap(value))
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Explanation comparison saved to {save_path}")
        
        return fig


def test_visualizer():
    """Test the visualization tools with dummy data"""
    print("Testing ExplanationVisualizer...")
    
    from .explanation import AttributionGraphExplanation
    
    # Create dummy explanation
    explanation = AttributionGraphExplanation(
        graph_id="test_graph_viz",
        true_label=1,
        predicted_label=1,
        prediction_confidence=0.89,
        edge_mask=torch.randn(100),
        node_mask=torch.randn(50, 9),
        important_edges=[(0, 1, 0.9), (2, 3, 0.8), (4, 5, 0.7), (6, 7, 0.6)],
        important_nodes=[(0, "influence_node", 0.95), (1, "activation_node", 0.85), (2, "layer_node", 0.75)],
        critical_features={"influence": 0.85, "activation": 0.72, "layer": 0.65, "ctx_idx": 0.45},
        subgraph_nodes=[0, 1, 2, 3, 4, 5, 6, 7],
        suspicious_patterns=["high_influence_concentration", "context_position_manipulation"],
        fidelity_plus=0.82,
        fidelity_minus=0.76,
        sparsity=0.15
    )
    
    # Create dummy original data
    num_nodes = 50
    x = torch.randn(num_nodes, 9)
    edge_index = torch.randint(0, num_nodes, (2, 100))
    
    try:
        visualizer = ExplanationVisualizer()
        
        # Test feature importance plot
        fig1 = visualizer.plot_feature_importance(explanation, figsize=(8, 5))
        plt.close(fig1)
        
        print("✅ Visualization tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False


if __name__ == "__main__":
    test_visualizer()