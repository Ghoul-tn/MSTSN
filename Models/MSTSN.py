import torch
import torch.nn as nn
import numpy as np
from Models.SAM.GCN import GCN
from Models.TEM.GRU import GRU

class MSTSN_Gambia(nn.Module):
    def __init__(self, adj_matrix, gcn_dim1=64, gcn_dim2=32, gru_dim=64, gru_layers=2):
        super().__init__()
        
        # Register adjacency matrix as buffer so it moves with model
        adj_tensor = torch.FloatTensor(adj_matrix.toarray() if hasattr(adj_matrix, 'toarray') 
                                    else adj_matrix)
        self.register_buffer('adj', adj_tensor)
        
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
        batch_size = x.size(0)
        
        # Get matching submatrix of adj for current batch
        adj_batch = self.adj[:batch_size, :batch_size]
        
        # Ensure adj is on same device as input
        adj_batch = adj_batch.to(x.device)
        
        # Spatial processing
        gcn_out = self.gcn(x, adj_batch)  # (batch_size, seq_len, gcn_dim2)
        
        # Temporal processing
        gru_out, _ = self.gru(gcn_out)  # (batch_size, seq_len, gru_dim)
        
        # Attention
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Regression
        output = self.regressor(attn_out[:, -1, :])  # (batch_size, 1)
        return output.squeeze(1)  # (batch_size,)
