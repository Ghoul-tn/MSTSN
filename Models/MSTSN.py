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
        # XLA-specific checkpointing control
        self.use_checkpoint = True
        self.checkpoint_kwargs = {
            'use_reentrant': False,
            'preserve_rng_state': True
        }

    def _checkpointed_forward(self, module, *args):
        """XLA-compatible checkpointing wrapper"""
        if not self.use_checkpoint or not self.training:
            return module(*args)
        
        # For TPU, we need to use xm.mark_step() with checkpointing
        def custom_forward(*inputs):
            output = module(*inputs)
            xm.mark_step()  # Important for XLA
            return output
            
        return checkpoint(custom_forward, *args, **self.checkpoint_kwargs)

    def forward(self, x):
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Spatial Processing
        spatial_out = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            processed = self._checkpointed_forward(self.spatial_processor, x_t)
            spatial_out.append(processed.unsqueeze(1))
        spatial_out = torch.cat(spatial_out, dim=1)
        
        # Temporal Processing
        temporal_in = spatial_out.reshape(-1, seq_len, 64)
        temporal_out = self._checkpointed_forward(self.temporal_processor, temporal_in)
        
        # Cross Attention
        spatial_feats = spatial_out.reshape(batch_size, -1, 64)
        temporal_feats = temporal_out.reshape(batch_size, -1, 64)
        fused = self.cross_attn(spatial_feats, temporal_feats)
        
        return self.regressor(fused.mean(dim=1)).squeeze(-1)
