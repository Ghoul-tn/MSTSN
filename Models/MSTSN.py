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
        # Spatial dims: 3 -> 64 -> 128 (no dimension conflicts)
        self.spatial_processor = SpatialProcessor(
            num_nodes=num_nodes,
            in_dim=3,
            hidden_dim=64,
            out_dim=128  # Must match temporal_processor input_dim
        )
        
        # Temporal processor expects 128-dim inputs
        self.temporal_processor = TemporalTransformer(
            input_dim=128,  # Must match spatial_processor out_dim
            num_heads=2,
            ff_dim=256,
            num_layers=2
        )
        
        # Cross-attention dims
        self.cross_attn = CrossAttention(embed_dim=128, num_heads=2)
        
        # Regressor (no dimension changes)
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Spatial Processing
        spatial_out = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # [batch, nodes, 3]
            x_t = self.spatial_processor(x_t)  # [batch, nodes, 128]
            spatial_out.append(x_t.unsqueeze(1))
        spatial_out = torch.cat(spatial_out, dim=1)  # [batch, seq_len, nodes, 128]
        print(f"Spatial out: {spatial_out.shape}")  # Should be [batch, seq_len, nodes, 128]
        # Temporal Processing
        temporal_in = spatial_out.reshape(batch_size * num_nodes, seq_len, 128)
        print(f"Temporal in: {temporal_in.shape}")  # Should be [batch*nodes, seq_len, 128]
        temporal_out = self.temporal_processor(temporal_in)  # [batch*nodes, seq_len, 128]
        
        # Cross-Attention
        spatial_feats = spatial_out.reshape(batch_size, -1, 128)
        temporal_feats = temporal_out.reshape(batch_size, -1, 128)
        fused = self.cross_attn(spatial_feats, temporal_feats)
        
        return self.regressor(fused.mean(dim=1)).squeeze(-1)
