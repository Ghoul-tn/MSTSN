import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                batch_first=True,
                activation='gelu'
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
