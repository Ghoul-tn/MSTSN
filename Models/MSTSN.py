import torch
import torch.nn as nn
from Models.SAM.GCN import SpatialProcessor
from Models.TEM.GRU import TemporalTransformer

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.spatial_query = nn.Linear(embed_dim, embed_dim)
        self.temporal_key = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, spatial_feats, temporal_feats):
        query = self.spatial_query(spatial_feats)
        key = self.temporal_key(temporal_feats)
        attn_out, _ = self.attn(query, key, value=temporal_feats)
        return attn_out

class EnhancedMSTSN(nn.Module):
    def __init__(self, num_nodes, gcn_dim1=128, gcn_dim2=256, transformer_dim=192, num_heads=4):
        super().__init__()
        # Spatial Module
        self.spatial_processor = SpatialProcessor(num_nodes, 3, gcn_dim1, gcn_dim2)
        
        # Temporal Module
        self.temporal_processor = TemporalTransformer(
            input_dim=gcn_dim2,
            num_heads=num_heads,
            ff_dim=transformer_dim * 4,
            num_layers=3
        )
        
        # Cross-Attention Fusion
        self.cross_attn = CrossAttention(embed_dim=gcn_dim2, num_heads=num_heads)
        
        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(gcn_dim2, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # Input shape: [batch, seq_len, nodes, features]
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Spatial Processing (per timestep)
        spatial_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # [batch, nodes, 3]
            x_t = self.spatial_processor(x_t)
            spatial_outputs.append(x_t.unsqueeze(1))
        spatial_out = torch.cat(spatial_outputs, dim=1)  # [batch, seq_len, nodes, gcn_dim2]
        
        # Temporal Processing
        temporal_in = spatial_out.reshape(batch_size * num_nodes, seq_len, -1)
        temporal_out = self.temporal_processor(temporal_in)  # [batch*nodes, seq_len, gcn_dim2]
        
        # Cross-Attention Fusion
        spatial_feats = spatial_out.reshape(batch_size, seq_len * num_nodes, -1)
        temporal_feats = temporal_out.reshape(batch_size, seq_len * num_nodes, -1)
        fused = self.cross_attn(spatial_feats, temporal_feats)
        
        # Final Prediction
        aggregated = fused.mean(dim=1)  # [batch, gcn_dim2]
        output = self.regressor(aggregated)  # [batch, 1]
        return output.squeeze(-1)
