import torch
import torch.nn as nn
from Models.SAM.GCN import GCN
from Models.TEM.GRU import GRU

class MSTSN_Gambia(nn.Module):
    def __init__(self, adj_matrix, gcn_dim1=128, gcn_dim2=64, gru_dim=128, gru_layers=2):
        super().__init__()
        
        # Store architecture parameters
        self.gcn_dim2 = gcn_dim2
        self.gru_dim = gru_dim
        self.num_nodes = adj_matrix.shape[0]  # Should be 2139
        
        # Convert and store adjacency matrix
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
            nn.Linear(gru_dim, 64),
            nn.GELU(),
            nn.Dropout(0.5),  # Increased from 0.2
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """x shape: (batch_size, seq_len, num_nodes, features)"""
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Process each timestep through GCN
        gcn_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # [batch, nodes, 3]
            gcn_out = self.gcn(x_t, self.adj)  # [batch, nodes, gcn_dim2]
            gcn_outputs.append(gcn_out.unsqueeze(1))  # Add seq dimension
            
        # Combine temporal outputs [batch, seq_len, nodes, gcn_dim2]
        gcn_out = torch.cat(gcn_outputs, dim=1)
        
        # Prepare for GRU: reshape to process each node's time series
        gru_in = gcn_out.permute(0, 2, 1, 3)  # [batch, nodes, seq_len, gcn_dim2]
        gru_in = gru_in.reshape(batch_size * num_nodes, seq_len, -1)
        
        # Temporal processing
        gru_out, _ = self.gru(gru_in)  # [batch*nodes, seq_len, gru_dim]
        gru_out = gru_out.reshape(batch_size, num_nodes, seq_len, -1)
        
        # Attention across time steps
        attn_in = gru_out.permute(0, 2, 1, 3)  # [batch, seq_len, nodes, gru_dim]
        attn_out, _ = self.attention(
            attn_in.reshape(batch_size * seq_len, num_nodes, -1),
            attn_in.reshape(batch_size * seq_len, num_nodes, -1),
            attn_in.reshape(batch_size * seq_len, num_nodes, -1)
        )
        
        # Final prediction
        output = self.regressor(attn_out)  # [batch*seq_len*nodes, 1]
        output = output.reshape(batch_size, seq_len, num_nodes)
        
        # Return predictions for last timestep
        return output[:, -1, :]  # [batch, nodes]
