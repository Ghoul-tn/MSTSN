import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from Models.SAM.GCN import SpatialProcessor
from Models.TEM.GRU import TemporalTransformer

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
            hidden_dim=32,  # Reduced from 64
            out_dim=64  # Reduced from 128
        )
        
        self.temporal_processor = TemporalTransformer(
            input_dim=64,  # Matches reduced spatial output
            num_heads=2,
            ff_dim=128,  # Reduced from 256
            num_layers=1  # Reduced from 2
        )
        
        # Smaller regressor
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        # Convert all parameters to bfloat16
        for param in self.parameters():
            param.data = param.data.to(torch.bfloat16)
        # Manual memory management
        self._spatial_outputs = None
        self._temporal_output = None

    def _spatial_forward(self, x_t):
        """Forward pass with manual memory management"""
        # Clear previous outputs
        if self._spatial_outputs is None:
            self._spatial_outputs = []
        
        # Forward pass with manual memory clearing
        with torch.no_grad():
            out = self.spatial_processor(x_t)
        self._spatial_outputs.append(out.unsqueeze(1))
        xm.mark_step()  # Important for XLA
        return out

    def _temporal_forward(self, x):
        """Temporal forward with manual memory management"""
        with torch.no_grad():
            out = self.temporal_processor(x)
        self._temporal_output = out
        xm.mark_step()
        return out

    def forward(self, x):
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Spatial Processing
        self._spatial_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            self._spatial_forward(x_t)
        spatial_out = torch.cat(self._spatial_outputs, dim=1)
        
        # Temporal Processing
        temporal_in = spatial_out.reshape(-1, seq_len, 64)
        self._temporal_forward(temporal_in)
        temporal_out = self._temporal_output
        
        # Cross Attention
        spatial_feats = spatial_out.reshape(batch_size, -1, 64)
        temporal_feats = temporal_out.reshape(batch_size, -1, 64)
        fused = self.cross_attn(spatial_feats, temporal_feats)
        
        # Final prediction
        output = self.regressor(fused.mean(dim=1)).squeeze(-1)
        
        # Clear intermediate values
        self._spatial_outputs = None
        self._temporal_output = None
        
        return output
