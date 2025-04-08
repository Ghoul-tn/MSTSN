import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.embed_dim = input_dim  # Store original dimension
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,  # Must match input_dim
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Ensure input dimension matches
        if x.size(-1) != self.embed_dim:
            raise ValueError(f"Input dim {x.size(-1)} != expected {self.embed_dim}")
        return self.norm(self.transformer(x))
