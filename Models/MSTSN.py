import torch.nn as nn
from Models.SAM.GCN import GCN
from Models.TEM.GRU import GRU

class MSTSN_Gambia(nn.Module):
    def __init__(self, adj_matrix):
        super().__init__()
        # Convert sparse adj matrix to dense for GCN
        adj_dense = adj_matrix.toarray() if hasattr(adj_matrix, 'toarray') else adj_matrix
        
        # Spatial Module (GNN)
        self.gcn = GCN(
            input_dim=3, 
            output_dim1=64, 
            output_dim2=32, 
            adj=adj_dense
        )
        
        # Temporal Module
        self.gru = GRU(
            feature_size=32,
            hidden_size=64,
            num_layers=2,
            output_size=32
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            dropout=0.1
        )
        
        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, 3)
        batch_size = x.shape[0]
        
        # Spatial processing
        x_spatial = x.permute(1, 0, 2)  # (seq_len, batch, 3)
        gcn_out = self.gcn(x_spatial)    # (seq_len, batch, 32)
        
        # Temporal processing
        gru_out = self.gru(gcn_out)      # (seq_len, batch, 32)
        attn_out, _ = self.attention(
            gru_out, gru_out, gru_out
        )
        
        # Regression
        output = self.regressor(attn_out[-1])
        return output.squeeze()
