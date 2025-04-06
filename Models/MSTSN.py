import torch.nn as nn
from Models.SAM.GCN import GCN
from Models.TEM.GRU import GRU

class MSTSN_Gambia(nn.Module):
    def __init__(self, adj_matrix, gcn_dim1=64, gcn_dim2=32, gru_dim=64, gru_layers=2):
        super().__init__()
        # Convert adj matrix to sparse tensor
        adj_coo = adj_matrix.tocoo()
        indices = torch.LongTensor(np.vstack((adj_coo.row, adj_coo.col)))
        values = torch.FloatTensor(adj_coo.data)
        shape = adj_coo.shape
        self.adj = torch.sparse.FloatTensor(indices, values, shape).to_dense()
        
        self.gcn = GCN(
            input_dim=3,  # NDVI, Soil, LST
            hidden_dim=gcn_dim1,
            output_dim=gcn_dim2,
            adj=self.adj
        )
        
        self.gru = GRU(
            feature_size=gcn_dim2,
            hidden_size=gru_dim,
            num_layers=gru_layers,
            output_size=gru_dim
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=gru_dim,
            num_heads=4
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(gru_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, 3)
        batch_size, seq_len, _ = x.shape
        
        # Reshape for GCN: (seq_len, batch_size, features)
        x = x.permute(1, 0, 2)
        
        # Spatial processing
        gcn_out = self.gcn(x)  # (seq_len, batch_size, gcn_dim2)
        
        # Temporal processing
        gru_out, _ = self.gru(gcn_out)  # (seq_len, batch_size, gru_dim)
        
        # Attention
        attn_out, _ = self.attention(
            gru_out, gru_out, gru_out
        )
        
        # Regression (use last timestep)
        output = self.regressor(attn_out[-1])
        return output.squeeze()
