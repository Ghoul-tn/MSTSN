import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x shape: [batch * nodes, seq_len, features]
        x = self.transformer(x)
        return self.norm(x)
