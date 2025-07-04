# GNNExplainer Technical Implementation for Circuit-Tracer GraphGPS

## Model Architecture Analysis

### Your Current GraphGPS Model
```python
# From your working model (models.py)
class PromptInjectionGraphGPS:
    def __init__(self, input_dim=9, hidden_dim=256, num_layers=6, num_heads=8, num_classes=2, dropout=0.1):
        # 1. Input projection: 9 features → 256 hidden
        self.input_proj = Linear(input_dim, hidden_dim)
        
        # 2. GraphGPS layers (6 layers with global attention)
        self.gps_layers = ModuleList([
            GraphGPSLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # 3. Global pooling: graph-level representation
        self.pooling = global_mean_pool
        
        # 4. Classification head: 256 → 2 classes
        self.classifier = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x, edge_index, batch, edge_attr=None):
        # Node embeddings through GPS layers
        h = self.input_proj(x)
        for layer in self.gps_layers:
            h = layer(h, edge_index, batch)
        
        # Graph-level prediction
        h_graph = self.pooling(h, batch)
        logits = self.classifier(h_graph)
        return logits, h  # Return both logits and node embeddings
```

### GNNExplainer Integration Points
The key insight: **GNNExplainer needs to hook into your model's forward pass to learn masks**

## Technical Implementation

### 1. GNNExplainer Core Algorithm

```python
# explainer/gnn_explainer_core.py
import torch
import torch.nn as nn
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks
from torch_geometric.utils import k_hop_subgraph, subgraph

class CircuitTracerGNNExplainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Configure GNNExplainer for graph classification
        self.algorithm = GNNExplainer(
            epochs=200,              # Optimization steps
            lr=0.01,                # Learning rate for mask optimization
            edge_size=0.005,        # L1 penalty for edge sparsity
            edge_ent=1.0,           # Entropy penalty for edge mask
            node_feat_size=1.0,     # L1 penalty for node feature sparsity
            node_feat_ent=0.1,      # Entropy penalty for node features
        )
        
        self.explainer = Explainer(
            model=model,
            algorithm=self.algorithm,
            explanation_type='model',        # Explain model predictions
            node_mask_type='attributes',     # Mask node features
            edge_mask_type='object',         # Mask individual edges
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',          # Graph-level classification
                return_type='probs',         # Return probabilities
            ),
        )
    
    def explain_graph(self, data, target=None):
        """
        Generate explanation for a single graph
        
        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch
            target: Optional target class (if None, use model prediction)
        
        Returns:
            Explanation object with masks and metadata
        """
        data = data.to(self.device)
        
        # Get model prediction if target not specified
        if target is None:
            with torch.no_grad():
                logits, _ = self.model(data.x, data.edge_index, data.batch, data.edge_attr)
                target = logits.argmax(dim=-1).item()
        
        # Generate explanation
        explanation = self.explainer(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            batch=data.batch,
            target=target
        )
        
        return explanation
```

### 2. Explanation Data Structure

```python
# explainer/explanation.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np

@dataclass
class AttributionGraphExplanation:
    """Explanation specifically for attribution graphs"""
    
    # Basic info
    graph_id: str
    true_label: int                    # 0=benign, 1=injected
    predicted_label: int
    prediction_confidence: float
    
    # Raw explanation masks from GNNExplainer
    edge_mask: torch.Tensor           # [num_edges] - importance of each edge
    node_mask: torch.Tensor           # [num_nodes, num_features] - feature importance
    
    # Processed explanations
    important_edges: List[Tuple[int, int, float]]  # [(src, dst, importance), ...]
    important_nodes: List[Tuple[int, str, float]]  # [(node_idx, node_id, importance), ...]
    critical_features: Dict[str, float]            # {feature_name: importance}
    
    # Subgraph extraction
    explanation_subgraph: Optional[torch.Tensor]   # Minimal explaining subgraph
    subgraph_nodes: List[int]                      # Node indices in subgraph
    
    # Domain-specific interpretations
    suspicious_patterns: List[str]                 # Identified attack patterns
    circuit_insights: Dict[str, any]               # Circuit-tracer specific insights
    
    # Quality metrics
    fidelity_plus: float                          # Necessity score
    fidelity_minus: float                         # Sufficiency score  
    sparsity: float                               # Explanation compactness
    
    def to_dict(self) -> dict:
        """Convert to serializable dictionary"""
        return {
            'graph_id': self.graph_id,
            'true_label': self.true_label,
            'predicted_label': self.predicted_label,
            'prediction_confidence': self.prediction_confidence,
            'important_edges': self.important_edges,
            'important_nodes': self.important_nodes,
            'critical_features': self.critical_features,
            'suspicious_patterns': self.suspicious_patterns,
            'fidelity_plus': self.fidelity_plus,
            'fidelity_minus': self.fidelity_minus,
            'sparsity': self.sparsity
        }
```

### 3. Domain-Specific Post-Processing

```python
# explainer/circuit_tracer_processor.py
class CircuitTracerExplanationProcessor:
    """Convert raw GNNExplainer output to circuit-tracer domain insights"""
    
    def __init__(self, converter, vocab_mapping):
        self.converter = converter
        self.vocab_mapping = vocab_mapping  # node_id -> human readable
        self.feature_names = [
            'influence', 'activation', 'layer', 'ctx_idx', 'feature',
            'is_cross_layer_transcoder', 'is_mlp_error', 'is_embedding', 'is_target_logit'
        ]
    
    def process_explanation(self, explanation, original_data) -> AttributionGraphExplanation:
        """Convert PyG explanation to attribution graph explanation"""
        
        # Extract important edges (top 10%)
        edge_importance = explanation.edge_mask
        num_important = max(1, int(0.1 * len(edge_importance)))
        top_edge_indices = torch.topk(edge_importance, num_important).indices
        
        important_edges = []
        for idx in top_edge_indices:
            src, dst = original_data.edge_index[:, idx].tolist()
            importance = edge_importance[idx].item()
            important_edges.append((src, dst, importance))
        
        # Extract important nodes (by aggregating feature importance)
        node_importance = explanation.node_mask.sum(dim=1)  # Sum across features
        num_important_nodes = max(1, int(0.1 * len(node_importance)))
        top_node_indices = torch.topk(node_importance, num_important_nodes).indices
        
        important_nodes = []
        for idx in top_node_indices:
            node_id = self.vocab_mapping.get(idx.item(), f"node_{idx}")
            importance = node_importance[idx].item()
            important_nodes.append((idx.item(), node_id, importance))
        
        # Analyze feature importance
        feature_importance = explanation.node_mask.mean(dim=0)  # Average across nodes
        critical_features = {
            name: importance.item() 
            for name, importance in zip(self.feature_names, feature_importance)
            if importance.item() > 0.1  # Threshold for significance
        }
        
        # Identify suspicious patterns
        suspicious_patterns = self._identify_attack_patterns(
            important_edges, important_nodes, critical_features
        )
        
        # Extract circuit insights
        circuit_insights = self._extract_circuit_insights(
            important_edges, important_nodes, original_data
        )
        
        return AttributionGraphExplanation(
            graph_id=f"graph_{id(original_data)}",
            true_label=original_data.y.item() if hasattr(original_data, 'y') else -1,
            predicted_label=explanation.prediction.argmax().item(),
            prediction_confidence=torch.softmax(explanation.prediction, dim=0).max().item(),
            edge_mask=edge_importance,
            node_mask=explanation.node_mask,
            important_edges=important_edges,
            important_nodes=important_nodes,
            critical_features=critical_features,
            suspicious_patterns=suspicious_patterns,
            circuit_insights=circuit_insights,
            fidelity_plus=0.0,  # Computed separately
            fidelity_minus=0.0, # Computed separately
            sparsity=len(top_edge_indices) / len(edge_importance)
        )
    
    def _identify_attack_patterns(self, edges, nodes, features):
        """Identify known prompt injection patterns"""
        patterns = []
        
        # Pattern 1: High influence concentration
        if features.get('influence', 0) > 0.5:
            patterns.append("high_influence_concentration")
        
        # Pattern 2: Context manipulation (high ctx_idx importance)
        if features.get('ctx_idx', 0) > 0.3:
            patterns.append("context_position_manipulation")
        
        # Pattern 3: Unusual layer patterns
        if features.get('layer', 0) > 0.4:
            patterns.append("cross_layer_attack")
        
        # Pattern 4: Target logit manipulation
        if features.get('is_target_logit', 0) > 0.2:
            patterns.append("direct_logit_manipulation")
        
        return patterns
    
    def _extract_circuit_insights(self, edges, nodes, data):
        """Extract circuit-tracer specific insights"""
        insights = {
            'attack_depth': 0,          # How deep the attack penetrates
            'affected_layers': set(),   # Which layers are involved
            'manipulation_type': '',    # Type of manipulation detected
            'confidence': 0.0          # Confidence in the explanation
        }
        
        # Analyze layer distribution
        # Extract from node features if available
        # This would require mapping back to original node data
        
        return insights
```

### 4. Evaluation Metrics Implementation

```python
# explainer/metrics.py
class ExplanationEvaluator:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def evaluate_explanation(self, explanation, original_data):
        """Compute fidelity and other metrics"""
        
        # Fidelity+ (Necessity): How much does prediction change when removing explanation?
        fidelity_plus = self._compute_fidelity_plus(explanation, original_data)
        
        # Fidelity- (Sufficiency): Can explanation alone maintain prediction?
        fidelity_minus = self._compute_fidelity_minus(explanation, original_data)
        
        # Sparsity: How compact is the explanation?
        sparsity = self._compute_sparsity(explanation)
        
        return {
            'fidelity_plus': fidelity_plus,
            'fidelity_minus': fidelity_minus,
            'sparsity': sparsity
        }
    
    def _compute_fidelity_plus(self, explanation, data):
        """Necessity: prediction change when removing important edges/nodes"""
        
        # Get original prediction
        with torch.no_grad():
            orig_logits, _ = self.model(data.x, data.edge_index, data.batch, data.edge_attr)
            orig_pred = torch.softmax(orig_logits, dim=-1)
        
        # Create masked version by removing important edges
        edge_mask = explanation.edge_mask
        threshold = torch.quantile(edge_mask, 0.9)  # Remove top 10% edges
        keep_edges = edge_mask < threshold
        
        masked_edge_index = data.edge_index[:, keep_edges]
        masked_edge_attr = data.edge_attr[keep_edges] if data.edge_attr is not None else None
        
        # Get prediction on masked graph
        with torch.no_grad():
            masked_logits, _ = self.model(data.x, masked_edge_index, data.batch, masked_edge_attr)
            masked_pred = torch.softmax(masked_logits, dim=-1)
        
        # Compute prediction change
        fidelity = (orig_pred - masked_pred).abs().max().item()
        return fidelity
    
    def _compute_fidelity_minus(self, explanation, data):
        """Sufficiency: prediction with only explanation subgraph"""
        
        # Extract explanation subgraph (top edges)
        edge_mask = explanation.edge_mask
        threshold = torch.quantile(edge_mask, 0.9)  # Keep top 10% edges
        keep_edges = edge_mask >= threshold
        
        if keep_edges.sum() == 0:
            return 0.0
        
        subgraph_edge_index = data.edge_index[:, keep_edges]
        subgraph_edge_attr = data.edge_attr[keep_edges] if data.edge_attr is not None else None
        
        # Get unique nodes in subgraph
        unique_nodes = torch.unique(subgraph_edge_index)
        
        # Create node mapping and remap edges
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(unique_nodes)}
        remapped_edges = torch.tensor([
            [node_mapping[src.item()], node_mapping[dst.item()]]
            for src, dst in subgraph_edge_index.t()
        ]).t()
        
        # Extract subgraph node features
        subgraph_x = data.x[unique_nodes]
        subgraph_batch = torch.zeros(len(unique_nodes), dtype=torch.long, device=data.x.device)
        
        # Get prediction on subgraph
        with torch.no_grad():
            sub_logits, _ = self.model(subgraph_x, remapped_edges, subgraph_batch, subgraph_edge_attr)
            sub_pred = torch.softmax(sub_logits, dim=-1)
        
        # Get original prediction
        with torch.no_grad():
            orig_logits, _ = self.model(data.x, data.edge_index, data.batch, data.edge_attr)
            orig_pred = torch.softmax(orig_logits, dim=-1)
        
        # Compute similarity
        fidelity = 1.0 - (orig_pred - sub_pred).abs().max().item()
        return max(0.0, fidelity)
    
    def _compute_sparsity(self, explanation):
        """Fraction of edges/nodes in explanation"""
        edge_mask = explanation.edge_mask
        threshold = torch.quantile(edge_mask, 0.9)
        important_edges = (edge_mask >= threshold).sum().item()
        total_edges = len(edge_mask)
        
        return important_edges / total_edges
```

### 5. Main Explanation Script

```python
# explain_model.py
#!/usr/bin/env python3
"""
Generate explanations for GraphGPS prompt injection predictions
"""

import torch
import argparse
from pathlib import Path
import pickle
import json

# Import your existing modules
try:
    from train.models import PromptInjectionGraphGPS
    from train.dataset import create_data_loaders
    from train.data_converter import AttributionGraphConverter
except ImportError:
    from models import PromptInjectionGraphGPS
    from dataset import create_data_loaders
    from data_converter import AttributionGraphConverter

# Import explainer modules
from explainer.gnn_explainer_core import CircuitTracerGNNExplainer
from explainer.circuit_tracer_processor import CircuitTracerExplanationProcessor
from explainer.metrics import ExplanationEvaluator

def load_trained_model(model_path, device='cuda'):
    """Load your trained GraphGPS model"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model config
    model_config = checkpoint.get('model_config', {
        'input_dim': 9,
        'hidden_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'num_classes': 2,
        'dropout': 0.1
    })
    
    # Initialize model
    model = PromptInjectionGraphGPS(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def explain_single_graph(model_path, data_path, output_dir):
    """Generate explanation for a single graph"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading trained model...")
    model = load_trained_model(model_path, device)
    
    # Load data (implement based on your data format)
    print("Loading graph data...")
    # This depends on your specific data format
    # You'll need to load a PyG Data object here
    
    # Initialize explainer
    print("Initializing explainer...")
    explainer = CircuitTracerGNNExplainer(model, device)
    
    # Initialize processor
    converter = AttributionGraphConverter()  # You'll need vocab loaded
    processor = CircuitTracerExplanationProcessor(converter, {})  # Add vocab mapping
    
    # Initialize evaluator
    evaluator = ExplanationEvaluator(model, device)
    
    # Generate explanation
    print("Generating explanation...")
    raw_explanation = explainer.explain_graph(data)
    
    # Process into domain-specific format
    processed_explanation = processor.process_explanation(raw_explanation, data)
    
    # Evaluate explanation quality
    metrics = evaluator.evaluate_explanation(raw_explanation, data)
    processed_explanation.fidelity_plus = metrics['fidelity_plus']
    processed_explanation.fidelity_minus = metrics['fidelity_minus']
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / 'explanation.json', 'w') as f:
        json.dump(processed_explanation.to_dict(), f, indent=2)
    
    print(f"Explanation saved to {output_path}")
    
    return processed_explanation

def explain_dataset_batch(model_path, dataset_path, output_dir, batch_size=32):
    """Generate explanations for entire test dataset"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_trained_model(model_path, device)
    
    # Load dataset (reuse your existing dataset loading)
    # You'll need to adapt this to load your test dataset
    
    # Initialize components
    explainer = CircuitTracerGNNExplainer(model, device)
    processor = CircuitTracerExplanationProcessor(converter, vocab_mapping)
    evaluator = ExplanationEvaluator(model, device)
    
    explanations = []
    
    # Process in batches
    for batch_idx, batch in enumerate(test_loader):
        print(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
        
        # Generate explanations for each graph in batch
        for i in range(batch.batch.max().item() + 1):
            # Extract single graph from batch
            mask = batch.batch == i
            single_graph = extract_single_graph(batch, mask)
            
            # Generate and process explanation
            raw_explanation = explainer.explain_graph(single_graph)
            processed_explanation = processor.process_explanation(raw_explanation, single_graph)
            
            # Evaluate
            metrics = evaluator.evaluate_explanation(raw_explanation, single_graph)
            processed_explanation.fidelity_plus = metrics['fidelity_plus']
            processed_explanation.fidelity_minus = metrics['fidelity_minus']
            
            explanations.append(processed_explanation)
    
    # Save batch results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save individual explanations
    for i, explanation in enumerate(explanations):
        with open(output_path / f'explanation_{i:04d}.json', 'w') as f:
            json.dump(explanation.to_dict(), f, indent=2)
    
    # Save summary statistics
    summary = compute_explanation_summary(explanations)
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Generated {len(explanations)} explanations in {output_path}")
    
    return explanations

def main():
    parser = argparse.ArgumentParser(description='Generate GNN explanations for prompt injection detection')
    parser.add_argument('--model_path', required=True, help='Path to trained GraphGPS model (.pt file)')
    parser.add_argument('--mode', choices=['single', 'batch'], default='single', help='Explanation mode')
    parser.add_argument('--data_path', help='Path to single graph data (for single mode)')
    parser.add_argument('--dataset_path', help='Path to dataset (for batch mode)')
    parser.add_argument('--output_dir', default='./explanations', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for batch mode')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.data_path:
            raise ValueError("--data_path required for single mode")
        explain_single_graph(args.model_path, args.data_path, args.output_dir)
    
    elif args.mode == 'batch':
        if not args.dataset_path:
            raise ValueError("--dataset_path required for batch mode")
        explain_dataset_batch(args.model_path, args.dataset_path, args.output_dir, args.batch_size)

if __name__ == "__main__":
    main()
```

### 6. Key Integration Points

#### Model Compatibility
Your GraphGPS model is **already compatible** with GNNExplainer because:
- ✅ It uses standard PyG operations (`Linear`, `global_mean_pool`)
- ✅ It returns logits in the expected format
- ✅ It follows PyG's forward signature

#### Critical Implementation Details

1. **Graph-Level vs Node-Level**: Your model does graph classification, so:
   ```python
   model_config=dict(
       task_level='graph',  # Not 'node'
       mode='multiclass_classification',
       return_type='probs'
   )
   ```

2. **Batch Handling**: GNNExplainer expects single graphs:
   ```python
   # Extract single graph from batch
   def extract_single_graph(batch_data, graph_idx):
       mask = batch_data.batch == graph_idx
       return Data(
           x=batch_data.x[mask],
           edge_index=subgraph_edge_index,  # Remapped edges
           edge_attr=batch_data.edge_attr[edge_mask] if batch_data.edge_attr is not None else None
       )
   ```

3. **Feature Mask Interpretation**: Your 9 features need specific handling:
   ```python
   feature_importance = explanation.node_mask.mean(dim=0)  # [9] tensor
   feature_names = ['influence', 'activation', 'layer', 'ctx_idx', 'feature', 
                   'is_cross_layer_transcoder', 'is_mlp_error', 'is_embedding', 'is_target_logit']
   ```

## Usage Example

```bash
# Explain a single graph
python explain_model.py \
    --model_path ./train_output/best_model.pt \
    --mode single \
    --data_path ./sample_graph.pkl \
    --output_dir ./explanations/single

# Explain entire test dataset  
python explain_model.py \
    --model_path ./train_output/best_model.pt \
    --mode batch \
    --dataset_path /home/sk2959/palmer_scratch/data/ \
    --output_dir ./explanations/batch \
    --batch_size 16
```

This technical implementation provides:
1. **Direct integration** with your existing GraphGPS model
2. **Domain-specific processing** for attribution graphs
3. **Proper evaluation metrics** (fidelity, sparsity)
4. **Production-ready scripts** for single and batch explanation
5. **Circuit-tracer specific insights** and pattern detection

The implementation accounts for your specific model architecture, graph structure, and domain requirements.