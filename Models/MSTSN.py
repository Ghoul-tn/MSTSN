import torch
import torch.nn as nn
import numpy as np
from Models.SAM.GCN import GCN
from Models.TEM.GRU import GRU

class MSTSN_Gambia(nn.Module):
    def __init__(self, adj_matrix, gcn_dim1=128, gcn_dim2=64, gru_dim=128, gru_layers=2):
        super().__init__()
        
        # Convert and register adjacency matrix
        adj_tensor = torch.FloatTensor(adj_matrix.toarray() if hasattr(adj_matrix, 'toarray') 
                                      else adj_matrix)
        self.register_buffer('adj', adj_tensor)
        
        # Spatial Module
        self.gcn = GCN(
            input_dim=3,
            hidden_dim=gcn_dim1,
            output_dim=gcn_dim2
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
        """x shape: (batch_size, seq_len, height, width, channels)"""
        batch_size, seq_len, h, w, _ = x.shape
        num_nodes = h * w
        
        # Reshape to graph format
        x = x.view(batch_size, seq_len, num_nodes, -1)  # (batch, seq_len, nodes, features)
        
        # Process each timestep
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # (batch, nodes, features)
            gcn_out = self.gcn(x_t, self.adj)  # (batch, nodes, gcn_dim2)
            outputs.append(gcn_out.unsqueeze(1))
            
        gcn_out = torch.cat(outputs, dim=1)  # (batch, seq_len, nodes, gcn_dim2)
        
        # Temporal processing
        gru_out, _ = self.gru(gcn_out)  # (batch, seq_len, num_nodes, gru_dim)
        
        # Regression
        output = self.regressor(gru_out[:, -1, :, :])  # (batch, num_nodes)
        return output
