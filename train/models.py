"""
GraphGPS model implementation for prompt injection detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import degree
from typing import Optional
import math

class PositionalEncoder(nn.Module):
    """Add positional encoding based on graph structure"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pos_proj = nn.Linear(3, hidden_dim)  # degree, node_index, random_walk
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        device = x.device
        num_nodes = x.size(0)
        
        # Calculate node degrees (normalized)
        if edge_index.size(1) > 0:
            row, col = edge_index
            deg = degree(row, num_nodes, dtype=x.dtype)
            deg = deg / (deg.max() + 1e-8)  # Normalize
        else:
            deg = torch.zeros(num_nodes, device=device, dtype=x.dtype)
        
        # Node index within graph (normalized)
        node_idx = torch.arange(num_nodes, device=device, dtype=x.dtype) / max(num_nodes - 1, 1)
        
        # Simple random walk feature (using degree as proxy)
        rw_feature = torch.sqrt(deg + 1e-8)
        
        # Create positional features
        pos_features = torch.stack([deg, node_idx, rw_feature], dim=1)
        
        # Project to hidden dimension
        pos_encoding = self.pos_proj(pos_features)
        
        return x + pos_encoding

class MultiHeadAttention(nn.Module):
    """Multi-head attention for graph nodes"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        batch_size = batch.max().item() + 1
        device = x.device
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # [num_nodes, hidden_dim]
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head attention
        Q = Q.view(-1, self.num_heads, self.head_dim)  # [num_nodes, num_heads, head_dim]
        K = K.view(-1, self.num_heads, self.head_dim)
        V = V.view(-1, self.num_heads, self.head_dim)
        
        # Apply attention within each graph in the batch
        output_list = []
        
        for b in range(batch_size):
            # Get nodes for this graph
            mask = (batch == b)
            if not mask.any():
                continue
                
            q_b = Q[mask]  # [num_nodes_b, num_heads, head_dim]
            k_b = K[mask]
            v_b = V[mask]
            
            # Compute attention scores
            scores = torch.einsum('nhd,mhd->nhm', q_b, k_b) / self.scale  # [num_nodes_b, num_heads, num_nodes_b]
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            out_b = torch.einsum('nhm,mhd->nhd', attn_weights, v_b)  # [num_nodes_b, num_heads, head_dim]
            output_list.append(out_b)
        
        if output_list:
            # Concatenate outputs
            output = torch.cat(output_list, dim=0)  # [num_nodes, num_heads, head_dim]
            output = output.view(-1, self.hidden_dim)  # [num_nodes, hidden_dim]
        else:
            output = torch.zeros_like(x)
        
        # Final projection
        output = self.out_proj(output)
        
        return output

class GraphGPSLayer(nn.Module):
    """Single GraphGPS layer combining local + global attention"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Local message passing (GCN)
        self.local_gnn = GCNConv(hidden_dim, hidden_dim)
        
        # Global multi-head attention
        self.global_attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Local message passing with residual connection
        x_local = self.local_gnn(x, edge_index, edge_weight)
        x = self.norm1(x + self.dropout(x_local))
        
        # 2. Global attention with residual connection
        x_global = self.global_attention(x, batch)
        x = self.norm2(x + self.dropout(x_global))
        
        # 3. Feed forward with residual connection
        x_ffn = self.ffn(x)
        x = self.norm3(x + self.dropout(x_ffn))
        
        return x

class GraphReadout(nn.Module):
    """Combine node embeddings into graph-level representation"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Combine mean, max, and sum pooling for richer representation
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        sum_pool = global_add_pool(x, batch)
        
        # Concatenate all pooling results
        return torch.cat([mean_pool, max_pool, sum_pool], dim=1)

class PromptInjectionGraphGPS(nn.Module):
    """Full GraphGPS model for prompt injection detection"""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int = 128, 
                 num_layers: int = 4, 
                 num_heads: int = 8, 
                 num_classes: int = 2, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding for graph structure
        self.pos_encoder = PositionalEncoder(hidden_dim)
        
        # GraphGPS layers
        self.gps_layers = nn.ModuleList([
            GraphGPSLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Graph-level readout (3x hidden_dim due to mean+max+sum pooling)
        self.readout = GraphReadout(hidden_dim)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> tuple:
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x, edge_index, batch)
        
        # Apply GraphGPS layers
        # Extract edge weights from edge_attr if provided
        edge_weight = None
        if edge_attr is not None and edge_attr.numel() > 0 and edge_attr.size(1) > 0:
            edge_weight = edge_attr[:, 0]
        
        for layer in self.gps_layers:
            x = layer(x, edge_index, batch, edge_weight)
        
        # Graph-level readout
        graph_embedding = self.readout(x, batch)
        
        # Final prediction
        logits = self.classifier(graph_embedding)
        
        return logits, x  # Return both logits and node embeddings for explainability

def test_model():
    """Test model with dummy data"""
    print("Testing PromptInjectionGraphGPS model...")
    
    # Create dummy data
    num_nodes = 20
    input_dim = 9
    hidden_dim = 64
    batch_size = 2
    
    # Node features
    x = torch.randn(num_nodes, input_dim)
    
    # Edges (create a simple graph)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 10, 11, 12, 13],  # Source nodes
        [1, 2, 3, 0, 11, 12, 13, 10]   # Target nodes
    ], dtype=torch.long)
    
    # Batch assignment (first 10 nodes to graph 0, next 10 to graph 1)
    batch = torch.tensor([0] * 10 + [1] * 10, dtype=torch.long)
    
    # Edge attributes
    edge_attr = torch.randn(edge_index.size(1), 1)
    
    # Create model
    model = PromptInjectionGraphGPS(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=3,
        num_heads=4,
        num_classes=2,
        dropout=0.1
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    try:
        with torch.no_grad():
            logits, node_embeddings = model(x, edge_index, batch, edge_attr)
        
        print(f"✅ Forward pass successful!")
        print(f"  Input shape: {x.shape}")
        print(f"  Output logits shape: {logits.shape}")
        print(f"  Node embeddings shape: {node_embeddings.shape}")
        print(f"  Batch size inferred: {batch.max().item() + 1}")
        
        # Test with different batch sizes
        batch2 = torch.tensor([0] * num_nodes, dtype=torch.long)  # Single graph
        logits2, _ = model(x, edge_index, batch2, edge_attr)
        print(f"  Single graph output shape: {logits2.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    test_model()