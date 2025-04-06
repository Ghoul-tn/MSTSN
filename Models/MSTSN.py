import torch
import torch.nn as nn
import numpy as np
from Models.SAM.GCN import GCN
from Models.TEM.GRU import GRU

class MSTSN_Gambia(nn.Module):
    def __init__(self, adj_matrix, gcn_dim1=128, gcn_dim2=64, gru_dim=128, gru_layers=2):
        super().__init__()
        self.adj_matrix = adj_matrix
        self.num_nodes = adj_matrix.shape[0]

        # Spatial Module
        self.gcn = GCN(
            input_dim=3,
            hidden_dim=gcn_dim1,
            output_dim=gcn_dim2,
            adj=adj_matrix
        )

        # Temporal Module
        self.gru = nn.GRU(
            input_size=gcn_dim2,
            hidden_size=gru_dim,
            num_layers=gru_layers,
            batch_first=True
        )

        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(gru_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_nodes)
        )

    def forward(self, x):
        """x shape: (batch, seq_len, num_nodes, 3)"""
        # Spatial processing
        batch_size, seq_len, num_nodes, _ = x.shape
        x = x.view(batch_size*seq_len, num_nodes, -1)
        gcn_out = self.gcn(x)  # (batch*seq_len, num_nodes, gcn_dim2)
        gcn_out = gcn_out.view(batch_size, seq_len, num_nodes, -1)
        
        # Temporal processing
        gru_out, _ = self.gru(gcn_out)  # (batch, seq_len, num_nodes, gru_dim)
        
        # Regression
        output = self.regressor(gru_out[:, -1, :, :])  # (batch, num_nodes)
        return output
