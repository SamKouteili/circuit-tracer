"""
Core GNNExplainer implementation for Circuit-Tracer GraphGPS model
"""

import torch
import torch.nn as nn
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.data import Data, Batch
from typing import Optional, Union, Dict, Any
import time

from explanation import AttributionGraphExplanation


class CircuitTracerGNNExplainer:
    """
    GNNExplainer specifically configured for Circuit-Tracer GraphGPS model
    
    This class wraps PyTorch Geometric's GNNExplainer with configuration
    optimized for graph-level prompt injection detection on attribution graphs.
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 device: str = 'cuda',
                 epochs: int = 200,
                 lr: float = 0.01,
                 edge_size: float = 0.005,
                 edge_ent: float = 1.0,
                 node_feat_size: float = 1.0,
                 node_feat_ent: float = 0.1):
        """
        Initialize Circuit-Tracer GNNExplainer
        
        Args:
            model: Trained GraphGPS model
            device: Device to run explanations on ('cuda' or 'cpu')
            epochs: Number of optimization epochs for explanation generation
            lr: Learning rate for mask optimization
            edge_size: L1 penalty coefficient for edge sparsity
            edge_ent: Entropy penalty coefficient for edge mask
            node_feat_size: L1 penalty coefficient for node feature sparsity
            node_feat_ent: Entropy penalty coefficient for node features
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()  # Set to evaluation mode
        
        # Store hyperparameters
        self.config = {
            'epochs': epochs,
            'lr': lr, 
            'edge_size': edge_size,
            'edge_ent': edge_ent,
            'node_feat_size': node_feat_size,
            'node_feat_ent': node_feat_ent
        }
        
        # Configure GNNExplainer algorithm with paper-compliant hyperparameters
        # Following original GNNExplainer paper (Ying et al., 2019):
        # - edge_size (α₁): 0.005 for edge sparsity regularization
        # - edge_ent (α₂): 1.0 for edge mask entropy regularization  
        # - node_feat_size (β₁): 1.0 for node feature sparsity regularization
        # - node_feat_ent (β₂): 0.1 for node feature entropy regularization
        self.algorithm = GNNExplainer(
            epochs=epochs,
            lr=lr,
            edge_size=edge_size,
            edge_ent=edge_ent,
            node_feat_size=node_feat_size,
            node_feat_ent=node_feat_ent,
        )
        
        # Configure main explainer for graph-level classification
        self.explainer = Explainer(
            model=model,
            algorithm=self.algorithm,
            explanation_type='model',        # Explain model predictions
            node_mask_type='attributes',     # Mask node features 
            edge_mask_type='object',         # Mask individual edges
            model_config=dict(
                mode='multiclass_classification',
                task_level='graph',          # Graph-level classification
                return_type='probs',         # Return class probabilities
            ),
        )
        
        print(f"CircuitTracerGNNExplainer initialized on {device}")
        print(f"Configuration: {self.config}")
    
    def explain_graph(self, 
                     data: Union[Data, Batch], 
                     target: Optional[int] = None,
                     index: Optional[int] = None) -> Any:
        """
        Generate explanation for a single graph
        
        Args:
            data: PyG Data object containing graph to explain
                 Must have x, edge_index, and batch attributes
                 Can optionally have edge_attr for edge weights
            target: Target class to explain (if None, use model prediction)
            index: Graph index if data is a batch (if None, assume single graph)
        
        Returns:
            PyG Explanation object with masks and predictions
        """
        start_time = time.time()
        
        # Ensure data is on correct device
        data = data.to(self.device)
        
        # Handle batch vs single graph
        if hasattr(data, 'batch'):
            if index is not None and data.batch.max().item() > 0:
                # Extract specific graph from batch
                data = self._extract_graph_from_batch(data, index)
            elif data.batch.max().item() > 0:
                raise ValueError("Data contains multiple graphs but no index specified")
        else:
            # Single graph - add batch dimension
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
        
        # Get model prediction if target not specified
        if target is None:
            with torch.no_grad():
                edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
                logits, _ = self.model(data.x, data.edge_index, data.batch, edge_attr)
                target = logits.argmax(dim=-1).item()
        
        # Generate explanation using PyG explainer
        try:
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
            explanation = self.explainer(
                x=data.x,
                edge_index=data.edge_index,
                edge_attr=edge_attr,
                batch=data.batch,
                target=target
            )
            
            # Add timing information
            explanation.explanation_time = time.time() - start_time
            
            return explanation
            
        except Exception as e:
            print(f"Error generating explanation: {e}")
            raise
    
    def _extract_graph_from_batch(self, batch_data: Batch, graph_idx: int) -> Data:
        """
        Extract a single graph from a batched Data object
        
        Args:
            batch_data: Batched PyG Data object
            graph_idx: Index of graph to extract
            
        Returns:
            Single Data object for the specified graph
        """
        # Find nodes belonging to the specified graph
        node_mask = batch_data.batch == graph_idx
        
        if not node_mask.any():
            raise ValueError(f"No nodes found for graph index {graph_idx}")
        
        # Extract node features
        x = batch_data.x[node_mask]
        
        # Extract edges involving nodes from this graph
        node_indices = torch.where(node_mask)[0]
        node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(node_indices)}
        
        # Find edges where both source and target are in this graph
        edge_mask = torch.isin(batch_data.edge_index[0], node_indices) & \
                   torch.isin(batch_data.edge_index[1], node_indices)
        
        if edge_mask.any():
            # Remap edge indices to new node numbering
            old_edges = batch_data.edge_index[:, edge_mask]
            new_edges = torch.tensor([
                [node_mapping[src.item()], node_mapping[dst.item()]]
                for src, dst in old_edges.t()
            ], device=batch_data.edge_index.device).t()
            
            # Extract corresponding edge attributes
            edge_attr = None
            if batch_data.edge_attr is not None:
                edge_attr = batch_data.edge_attr[edge_mask]
        else:
            # No edges in this graph
            new_edges = torch.empty((2, 0), dtype=torch.long, device=batch_data.edge_index.device)
            edge_attr = None
        
        # Create new Data object
        return Data(
            x=x,
            edge_index=new_edges,
            edge_attr=edge_attr,
            batch=torch.zeros(x.size(0), dtype=torch.long, device=x.device),
            y=batch_data.y[graph_idx] if hasattr(batch_data, 'y') else None
        )
    
    def explain_batch(self, 
                     batch_data: Batch, 
                     targets: Optional[torch.Tensor] = None) -> List[Any]:
        """
        Generate explanations for all graphs in a batch
        
        Args:
            batch_data: Batched PyG Data object
            targets: Optional target classes for each graph
            
        Returns:
            List of explanation objects, one per graph
        """
        num_graphs = batch_data.batch.max().item() + 1
        explanations = []
        
        if targets is not None and len(targets) != num_graphs:
            raise ValueError(f"Number of targets ({len(targets)}) doesn't match number of graphs ({num_graphs})")
        
        print(f"Generating explanations for {num_graphs} graphs...")
        
        for i in range(num_graphs):
            target = targets[i].item() if targets is not None else None
            
            try:
                explanation = self.explain_graph(batch_data, target=target, index=i)
                explanations.append(explanation)
                
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{num_graphs} explanations")
                    
            except Exception as e:
                print(f"  Failed to explain graph {i}: {e}")
                explanations.append(None)
        
        print(f"Generated {sum(1 for exp in explanations if exp is not None)}/{num_graphs} explanations")
        return explanations
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model being explained"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': type(self.model).__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': self.device,
            'explainer_config': self.config
        }
    
    def __repr__(self) -> str:
        return (f"CircuitTracerGNNExplainer("
                f"model={type(self.model).__name__}, "
                f"device={self.device}, "
                f"epochs={self.config['epochs']})")


def test_explainer():
    """Test the explainer with dummy data"""
    print("Testing CircuitTracerGNNExplainer...")
    
    # Create dummy model (simplified for testing)
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
    
    # Create dummy data
    num_nodes = 50
    x = torch.randn(num_nodes, 9)
    edge_index = torch.randint(0, num_nodes, (2, 100))
    edge_attr = torch.randn(100, 1)
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
    # Initialize explainer
    model = DummyModel()
    explainer = CircuitTracerGNNExplainer(model, device='cpu')
    
    # Test explanation
    try:
        explanation = explainer.explain_graph(data)
        print("✅ Explanation generated successfully")
        print(f"  Edge mask shape: {explanation.edge_mask.shape}")
        print(f"  Node mask shape: {explanation.node_mask.shape}")
        print(f"  Explanation time: {explanation.explanation_time:.3f}s")
        return True
    except Exception as e:
        print(f"❌ Explanation failed: {e}")
        return False


if __name__ == "__main__":
    test_explainer()