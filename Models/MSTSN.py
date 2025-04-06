import torch.nn as nn
from Models.SAM.GCN import GCN
from Models.TEM.GRU import GRU

class MSTSN_Gambia(nn.Module):
    def __init__(self, adj_matrix, gcn_dim1=64, gcn_dim2=32, gru_dim=64, gru_layers=2):
        super().__init__()
        
        # Store config
        self.gcn_dim1 = gcn_dim1
        self.gcn_dim2 = gcn_dim2
        self.gru_dim = gru_dim
        self.gru_layers = gru_layers
        
        # Convert sparse adj matrix to dense if needed
        adj_dense = adj_matrix.toarray() if hasattr(adj_matrix, 'toarray') else adj_matrix
        
        # Spatial Module (GNN)
        self.gcn = GCN(
            input_dim=3,  # NDVI, Soil, LST
            output_dim1=gcn_dim1,
            output_dim2=gcn_dim2,
            adj=adj_dense
        )
        
        # Temporal Module
        self.gru = GRU(
            feature_size=gcn_dim2,
            hidden_size=gru_dim,
            num_layers=gru_layers,
            output_size=gru_dim  # Same as hidden size
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=gru_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(gru_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, 3)
        batch_size = x.shape[0]
        
        # Spatial processing
        x_spatial = x.permute(1, 0, 2)  # (seq_len, batch, 3)
        gcn_out = self.gcn(x_spatial)    # (seq_len, batch, gcn_dim2)
        
        # Temporal processing
        gru_out = self.gru(gcn_out)      # (seq_len, batch, gru_dim)
        attn_out, _ = self.attention(
            gru_out, gru_out, gru_out
        )
        
        # Regression
        output = self.regressor(attn_out[-1])  # Use last timestep
        return output.squeeze()
