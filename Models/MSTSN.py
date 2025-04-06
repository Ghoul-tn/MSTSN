import torch
import torch.nn as nn
from Models.SAM.GCN import GCN
from Models.TEM.GRU import GRU

class MSTSN_Gambia(nn.Module):
    def __init__(self, adj_matrix, gcn_dim1=128, gcn_dim2=64, gru_dim=128, gru_layers=2):
        super().__init__()
        
        # Store adjacency matrix and calculate num_nodes
        adj_tensor = torch.FloatTensor(adj_matrix.toarray() if hasattr(adj_matrix, 'toarray') 
                                      else adj_matrix)
        self.register_buffer('adj', adj_tensor)
        self.num_nodes = adj_tensor.shape[0]  # Define num_nodes here
        
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
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Predict single value per node
        )

    def forward(self, x):
        """x shape: (batch_size, seq_len, height, width, channels)"""
        batch_size, seq_len, h, w, _ = x.shape
        
        # Reshape to node features (batch_size, seq_len, num_nodes, 3)
        x = x.reshape(batch_size, seq_len, h*w, 3)
        
        # Process each timestep
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # (batch_size, num_nodes, 3)
            gcn_out = self.gcn(x_t, self.adj)  # (batch_size, num_nodes, gcn_dim2)
            outputs.append(gcn_out.unsqueeze(1))
        
        # Stack temporal outputs (batch_size, seq_len, num_nodes, gcn_dim2)
        gcn_out = torch.cat(outputs, dim=1)
        
        # Temporal processing (flatten nodes for GRU)
        gru_in = gcn_out.reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, num_nodes*gcn_dim2)
        gru_out, _ = self.gru(gru_in)
        
        # Attention
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Regression (predict for each node)
        node_features = attn_out[:, -1, :].reshape(batch_size, self.num_nodes, -1)
        output = self.regressor(node_features)  # (batch_size, num_nodes, 1)
        
        return output.squeeze(-1)  # (batch_size, num_nodes)
