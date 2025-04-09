import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
from Models.SAM.GCN import SpatialProcessor
from Models.TEM.GRU import TemporalTransformer
from torch.utils.checkpoint import checkpoint

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        return self.attn(q, k, x2)[0]

class EnhancedMSTSN(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        # Reduced dimensions
        self.spatial_processor = SpatialProcessor(
            num_nodes=num_nodes,
            in_dim=3,
            hidden_dim=16,  # Reduced from 64
            out_dim=32  # Reduced from 128
        )
        
        self.temporal_processor = TemporalTransformer(
            input_dim=32,  # Matches reduced spatial output
            num_heads=2,
            ff_dim=64,  # Reduced from 256
            num_layers=1  # Reduced from 2
        )
        # Cross-attention dims
        self.cross_attn = CrossAttention(embed_dim=32, num_heads=2)
        # Smaller regressor
        self.regressor = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1)
        )


        self.use_checkpoint = True
    def _forward_impl(self, x):
        # Input: [batch, seq_len, nodes, features]
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Spatial Processing
        spatial_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            out = self.spatial_processor(x_t)  # Shape: [batch, nodes, 64]
            spatial_outputs.append(out.unsqueeze(1))
        
        spatial_out = torch.cat(spatial_outputs, dim=1)  # Shape: [batch, seq_len, nodes, 64]
        
        # Temporal Processing - FIXED RESHAPE
        # Reshape to [batch*nodes, seq_len, 64] for temporal processing
        temporal_in = spatial_out.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, 64)
        temporal_out = self.temporal_processor(temporal_in)  # Shape: [batch*nodes, seq_len, 64]
        
        # Reshape back to [batch, nodes, seq_len, 64]
        temporal_out = temporal_out.reshape(batch_size, num_nodes, seq_len, 64)
        
        # Cross Attention - FIXED DIMENSIONS
        # Average across sequence dimension for both
        spatial_feats = spatial_out.mean(dim=1)  # Shape: [batch, nodes, 64]
        temporal_feats = temporal_out.mean(dim=2)  # Shape: [batch, nodes, 64]
        
        # Now both have shape [batch, nodes, 64], perform cross-attention
        fused = self.cross_attn(spatial_feats, temporal_feats)  # Shape: [batch, nodes, 64]
        
        # Apply regressor to each node individually
        # Output should be [batch, nodes]
        node_preds = self.regressor(fused).squeeze(-1)
        
        return node_preds

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)
