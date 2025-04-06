import torch
import torch.nn as nn
import numpy as np
from Models.SAM.GCN import GCNBlock
from Models.TEM.GRU import GRU

class MSTSN_Gambia(nn.Module):
    def __init__(self, adj_matrix, gcn_dim1=128, gcn_dim2=64, gru_dim=128, gru_layers=2):
        super().__init__()
        
        # Store adjacency matrix
        self.adj_matrix = adj_matrix
        
        # Enhanced Spatial Module
        self.gcn_blocks = nn.Sequential(
            GCNBlock(3, gcn_dim1, adj_matrix),
            nn.Dropout(0.3),
            GCNBlock(gcn_dim1, gcn_dim2, adj_matrix)
        )
        
        # Temporal Module
        self.gru = nn.GRU(
            input_size=gcn_dim2,
            hidden_size=gru_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.2 if gru_layers > 1 else 0
        )
        self.temporal_norm = nn.LayerNorm(gru_dim)
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=gru_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(gru_dim, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, height, width, features)
        batch_size, seq_len, h, w, _ = x.shape
        
        # Reshape for GCN: (batch_size, seq_len, num_nodes, features)
        x = x.view(batch_size, seq_len, -1, 3)  # num_nodes = h * w
        
        # Spatial processing
        gcn_out = self.gcn_blocks(x)  # (batch_size, seq_len, num_nodes, gcn_dim2)
        
        # Reshape for GRU: (batch_size, seq_len, num_nodes * gcn_dim2)
        gru_in = gcn_out.view(batch_size, seq_len, -1)
        
        # Temporal processing
        gru_out, _ = self.gru(gru_in)
        gru_out = self.temporal_norm(gru_out)
        
        # Attention
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Combine features
        combined = gru_out + attn_out
        
        # Regression
        output = self.regressor(combined[:, -1, :])
        return output.squeeze()
