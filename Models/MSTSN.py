import torch
import torch.nn as nn
import numpy as np
from Models.SAM.GCN import GCN
from Models.TEM.GRU import GRU

class MSTSN_Gambia(nn.Module):
    def __init__(self, adj_matrix, gcn_dim1=64, gcn_dim2=32, gru_dim=64, gru_layers=2):
        super().__init__()
        
        # Store adjacency matrix as a dense tensor
        self.adj = torch.FloatTensor(adj_matrix.toarray() if hasattr(adj_matrix, 'toarray') 
                                    else adj_matrix)
        
        # Spatial Module
        self.gcn = GCN(
            input_dim=3,  # NDVI, Soil, LST
            hidden_dim=gcn_dim1,
            output_dim=gcn_dim2
        )
        
        # Temporal Module
        self.gru = GRU(
            input_size=gcn_dim2,
            hidden_size=gru_dim,
            num_layers=gru_layers
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=gru_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(gru_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, 3)
        batch_size = x.size(0)
        
        # Spatial processing
        adj_batch = self.adj[:batch_size, :batch_size]  # Match batch size
        gcn_out = self.gcn(x, adj_batch)  # (batch_size, seq_len, gcn_dim2)
        
        # Temporal processing
        gru_out, _ = self.gru(gcn_out)  # (batch_size, seq_len, gru_dim)
        
        # Attention
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Regression (use last timestep)
        output = self.regressor(attn_out[:, -1, :])  # (batch_size, 1)
        return output.squeeze(1)  # (batch_size,)
