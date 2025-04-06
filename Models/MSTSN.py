import torch
import torch.nn as nn
import numpy as np
from Models.SAM.GCN import GCN
from Models.TEM.GRU import GRU

class MSTSN_Gambia(nn.Module):
    def __init__(self, adj_matrix, gcn_dim1=128, gcn_dim2=64, gru_dim=128, gru_layers=3):
        super().__init__()
        
        # Enhanced GCN with residual connections
        self.gcn = nn.Sequential(
            GCNBlock(3, gcn_dim1, adj_matrix),
            nn.Dropout(0.3),
            GCNBlock(gcn_dim1, gcn_dim2, adj_matrix)
        )
        
        # Deeper GRU with layer normalization
        self.gru = nn.GRU(
            input_size=gcn_dim2,
            hidden_size=gru_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.2 if gru_layers > 1 else 0
        )
        self.gru_norm = nn.LayerNorm(gru_dim)
        
        # Multi-scale attention
        self.attention = nn.MultiheadAttention(
            embed_dim=gru_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Enhanced regression head
        self.regressor = nn.Sequential(
            nn.Linear(gru_dim, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Spatial processing
        gcn_out = self.gcn(x)  # [batch, seq_len, gcn_dim2]
        
        # Temporal processing
        gru_out, _ = self.gru(gcn_out)
        gru_out = self.gru_norm(gru_out)
        
        # Attention with skip connection
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        combined = gru_out + 0.5*attn_out  # Residual
        
        # Regression
        return self.regressor(combined[:, -1, :]).squeeze()
