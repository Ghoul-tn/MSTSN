import torch
import torch.nn as nn

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
        self.spatial_processor = SpatialProcessor(
            num_nodes=num_nodes,
            in_dim=3,
            hidden_dim=64,
            out_dim=128
        )
        self.temporal_processor = TemporalTransformer(
            input_dim=128,
            num_heads=2,
            ff_dim=256,
            num_layers=2
        )
        self.cross_attn = CrossAttention(embed_dim=128, num_heads=2)
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
            x_t = x[:, t, :, :]
            spatial_out.append(self.spatial_processor(x_t).unsqueeze(1))
        spatial_out = torch.cat(spatial_out, dim=1)  # [batch, seq_len, nodes, 128]
        
        # Temporal Processing
        temporal_in = spatial_out.reshape(batch_size * num_nodes, seq_len, -1)
        temporal_out = self.temporal_processor(temporal_in)
        
        # Cross-Attention
        spatial_feats = spatial_out.reshape(batch_size, -1, 128)
        temporal_feats = temporal_out.reshape(batch_size, -1, 128)
        fused = self.cross_attn(spatial_feats, temporal_feats)
        
        return self.regressor(fused.mean(dim=1)).squeeze(-1)
