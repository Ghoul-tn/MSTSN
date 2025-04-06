# 定义GRU网络
from torch import nn


class GRU(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False  # We use (seq_len, batch, features)
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (seq_len, batch_size, feature_size)
        output, _ = self.gru(x)  # output: (seq_len, batch_size, hidden_size)
        output = self.fc(output)
        return output, None  # Maintain compatibility
