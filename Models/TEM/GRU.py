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
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            bidirectional=True, batch_first=True, dropout=0.3
        )
        self.attn = nn.MultiheadAttention(2*hidden_size, 4, dropout=0.2)
        self.norm = nn.LayerNorm(2*hidden_size)

    def forward(self, x):
        out, _ = self.gru(x)  # [batch, seq, 2*hidden]
        out = out.permute(1, 0, 2)  # [seq, batch, 2*hidden]
        attn_out, _ = self.attn(out, out, out)
        return self.norm(out + attn_out).permute(1, 0, 2)
