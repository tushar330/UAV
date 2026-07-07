"""
Graph Encoder with Multi-Head Self-Attention.

Implements the ATOM encoder architecture from Paper A Section III-B:
1. GraphEmbeddingLayer: Linear projection of node features to D_hidden.
2. MultiHeadSelfAttention: Scaled dot-product attention with H heads.
3. GraphAttentionLayer: MHA + LayerNorm + FFN + residual connections.
4. GraphEncoder: Stacks L attention layers. Outputs per-node embeddings
   h^L_i ∈ R^D and graph summary h_sa ∈ R^D (mean pooling).

Supports both 2D input (3 features: [x, y, D]) and 3D input (4 features: [x, y, z, D]).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class GraphEmbeddingLayer(nn.Module):
    """
    Initial linear projection of node features to embedding space.

    Projects raw node features [x, y, (z), D] into a D_hidden-dimensional
    embedding vector for each node.

    Args:
        input_dim: Number of input features per node (3 for 2D, 4 for 3D).
        embed_dim: Output embedding dimension D_hidden.
    """

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, N, input_dim) raw node features.

        Returns:
            (batch_size, N, embed_dim) node embeddings.
        """
        return self.projection(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Scaled Dot-Product Self-Attention.

    Implements standard transformer-style attention:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Split across H attention heads, each with dimension d_k = D_hidden / H.

    Args:
        embed_dim: Total embedding dimension D_hidden.
        num_heads: Number of attention heads H.
        dropout: Dropout rate for attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Self-attention over node embeddings.

        Args:
            x: (batch_size, N, embed_dim) input embeddings.
            mask: Optional (batch_size, N) boolean mask. True = masked out.

        Returns:
            output: (batch_size, N, embed_dim) attended embeddings.
            attention_weights: (batch_size, H, N, N) attention weight matrix.
        """
        B, N, D = x.shape

        # Project to Q, K, V and reshape for multi-head
        # (B, N, D) → (B, N, H, d_k) → (B, H, N, d_k)
        Q = self.W_q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: (B, H, N, d_k) × (B, H, d_k, N) → (B, H, N, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            # mask: (B, N) → (B, 1, 1, N) for broadcasting
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            scores = scores.masked_fill(mask_expanded, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values: (B, H, N, N) × (B, H, N, d_k) → (B, H, N, d_k)
        attended = torch.matmul(attention_weights, V)

        # Reshape back: (B, H, N, d_k) → (B, N, H, d_k) → (B, N, D)
        attended = attended.transpose(1, 2).contiguous().view(B, N, D)

        # Output projection
        output = self.W_o(attended)

        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = ReLU(xW1 + b1)W2 + b2

    Args:
        embed_dim: Input and output dimension.
        ff_dim: Hidden dimension of the FFN.
        dropout: Dropout rate.
    """

    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, N, embed_dim).

        Returns:
            (batch_size, N, embed_dim).
        """
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention Layer: MHA + Add&Norm + FFN + Add&Norm.

    Follows the standard transformer encoder layer architecture:
        x' = LayerNorm(x + MHA(x))
        x'' = LayerNorm(x' + FFN(x'))

    Args:
        embed_dim: Embedding dimension D_hidden.
        num_heads: Number of attention heads H.
        ff_dim: Feed-forward hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, N, embed_dim).
            mask: Optional (batch_size, N) boolean mask.

        Returns:
            output: (batch_size, N, embed_dim).
            attention_weights: (batch_size, H, N, N).
        """
        # Multi-head self-attention with residual + norm
        attended, attn_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attended))

        # Feed-forward with residual + norm
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x, attn_weights


class GraphEncoder(nn.Module):
    """
    Full Graph Encoder: Embedding + L Self-Attention Layers + Pooling.

    Architecture (Paper A Section III-B):
    1. Linear embedding of raw node features → D_hidden
    2. L stacked GraphAttentionLayers (self-attention + FFN)
    3. Graph-level summary via mean pooling → h_sa

    Outputs:
        h_nodes: (B, N, D_hidden) — per-node embeddings h^L_i
        h_graph: (B, D_hidden) — graph summary embedding h_sa

    Args:
        input_dim: Number of raw features per node (3 for 2D, 4 for 3D).
        embed_dim: Embedding dimension D_hidden.
        num_heads: Number of attention heads H.
        ff_dim: Feed-forward hidden dimension.
        num_layers: Number of attention layers L.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 3,
        embed_dim: int = 512,
        num_heads: int = 8,
        ff_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        # Initial embedding
        self.embedding = GraphEmbeddingLayer(input_dim, embed_dim)

        # Stack of attention layers
        self.layers = nn.ModuleList([
            GraphAttentionLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        node_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode IoT node features into embeddings.

        Args:
            node_features: (batch_size, N, input_dim) raw node features.
            mask: Optional (batch_size, N) boolean mask for padded nodes.

        Returns:
            h_nodes: (batch_size, N, embed_dim) per-node embeddings.
            h_graph: (batch_size, embed_dim) graph summary (mean pooling).
        """
        # Initial linear embedding
        h = self.embedding(node_features)  # (B, N, D)

        # Pass through L attention layers
        all_attention_weights = []
        for layer in self.layers:
            h, attn_weights = layer(h, mask)
            all_attention_weights.append(attn_weights)

        # Graph-level summary via mean pooling
        if mask is not None:
            # Zero out masked positions before averaging
            mask_expanded = (~mask).unsqueeze(-1).float()  # (B, N, 1)
            h_masked = h * mask_expanded
            h_graph = h_masked.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            h_graph = h.mean(dim=1)  # (B, D)

        return h, h_graph

    def get_attention_weights(
        self,
        node_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> list:
        """
        Get attention weights from all layers (for visualization).

        Returns:
            List of L attention weight tensors, each (B, H, N, N).
        """
        h = self.embedding(node_features)
        weights = []
        for layer in self.layers:
            h, attn = layer(h, mask)
            weights.append(attn.detach())
        return weights


class GNNEncoder(nn.Module):
    """
    Plain message-passing GNN encoder (no multi-head self-attention).

    Ablation baseline for the "attention helps" experiment. Crucially this is a
    *trainable* module with the same input/output signature as GraphEncoder, so
    it can be trained with the identical TENMA loop and compared fairly against
    the attention encoder (instead of being evaluated with random weights).

    Architecture: per-node MLP + L rounds of mean-neighbour aggregation, then
    mean pooling for the graph summary h_sa.

    Args mirror GraphEncoder so it is a drop-in replacement.
    """

    def __init__(
        self,
        input_dim: int = 4,
        embed_dim: int = 512,
        num_heads: int = 8,   # unused, kept for signature compatibility
        ff_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_dim, embed_dim)
        # One aggregation block per layer: combines self embedding + neighbour mean.
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * embed_dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim),
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

    def forward(
        self,
        node_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.embedding(node_features))  # (B, N, D)

        for layer, norm in zip(self.layers, self.norms):
            if mask is not None:
                keep = (~mask).unsqueeze(-1).float()
                neigh = (h * keep).sum(dim=1, keepdim=True) / keep.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                neigh = h.mean(dim=1, keepdim=True)
            neigh = neigh.expand(-1, h.size(1), -1)
            update = layer(torch.cat([h, neigh], dim=-1))
            h = norm(h + update)

        if mask is not None:
            keep = (~mask).unsqueeze(-1).float()
            h_graph = (h * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
        else:
            h_graph = h.mean(dim=1)

        return h, h_graph
