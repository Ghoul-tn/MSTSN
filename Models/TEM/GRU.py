# 定义GRU网络
from torch import nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
    def forward(self, x):
        output, hidden = self.gru(x)
        return output, hidden

class EnhancedGRU(nn.Module):
    """Bidirectional GRU with Layer Normalization"""
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.norm = nn.LayerNorm(2*hidden_size)
        
    def forward(self, x):
        output, _ = self.gru(x)
        return self.norm(output)
