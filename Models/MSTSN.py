import torch
import torch.nn as nn
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

    def forward(self, x):
        x = x.to(torch.bfloat16)
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Spatial Processing with checkpointing
        spatial_out = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            if self.spatial_processor.checkpoint:
                x_t = checkpoint(self.spatial_processor, x_t)
            else:
                x_t = self.spatial_processor(x_t)
            spatial_out.append(x_t.unsqueeze(1))
        spatial_out = torch.cat(spatial_out, dim=1)
        
        # Temporal Processing with checkpointing
        temporal_in = spatial_out.reshape(-1, seq_len, spatial_out.size(-1))
        if self.temporal_processor.checkpoint:
            temporal_out = checkpoint(self.temporal_processor, temporal_in)
        else:
            temporal_out = self.temporal_processor(temporal_in)
        
        # Cross-Attention
        spatial_feats = spatial_out.reshape(batch_size, -1, spatial_out.size(-1))
        temporal_feats = temporal_out.reshape(batch_size, -1, temporal_out.size(-1))
        fused = self.cross_attn(spatial_feats, temporal_feats)
        return self.regressor(fused.mean(dim=1)).squeeze(-1)
