import torch
import torch.nn as nn
import numpy as np
from Models.SAM.GCN import GCN
from Models.TEM.GRU import EnhancedGRU

class MSTSN_Gambia(nn.Module):
    def __init__(self, adj_matrix, gcn_dim1=128, gcn_dim2=256, gru_dim=192, gru_layers=3):
        super().__init__()
        
        # Store architecture parameters
        self.num_nodes = adj_matrix.shape[0]
        self.gru_dim = gru_dim
        self.gru_layers = gru_layers
        
        # Convert and register adjacency matrix
        adj_tensor = torch.FloatTensor(adj_matrix.toarray() if hasattr(adj_matrix, 'toarray') 
                                     else adj_matrix)
        self.register_buffer('adj', adj_tensor)
        
        # Enhanced Spatial Module
        self.gcn = nn.Sequential(
            GCN(input_dim=3, hidden_dim=gcn_dim1, output_dim=gcn_dim2),
            nn.Dropout(0.3),
            nn.LayerNorm(gcn_dim2)
        )
        
        # Enhanced Temporal Module with Bidirectional GRU and Attention
        self.temporal_processor = EnhancedGRU(
            input_size=gcn_dim2,
            hidden_size=gru_dim,
            num_layers=gru_layers
        )
        
        # Multi-scale Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=2*gru_dim,  # Bidirectional
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        
        # Regression Head with Enhanced Capacity
        self.regressor = nn.Sequential(
            nn.Linear(2*gru_dim, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """Input shape: (batch_size, seq_len, num_nodes, features)"""
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Spatial Processing
        gcn_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # [batch, nodes, 3]
            gcn_out = self.gcn((x_t, self.adj))  # Modified GCN interface
            gcn_outputs.append(gcn_out.unsqueeze(1))
        
        gcn_out = torch.cat(gcn_outputs, dim=1)  # [batch, seq_len, nodes, gcn_dim2]
        
        # Temporal Processing
        gru_in = gcn_out.permute(0, 2, 1, 3)  # [batch, nodes, seq_len, gcn_dim2]
        gru_in = gru_in.reshape(batch_size*num_nodes, seq_len, -1)
        
        temporal_out = self.temporal_processor(gru_in)  # [batch*nodes, seq_len, 2*gru_dim]
        
        # Attention across time steps
        attn_in = temporal_out.reshape(batch_size, num_nodes, seq_len, -1)
        attn_in = attn_in.permute(0, 2, 1, 3)  # [batch, seq_len, nodes, 2*gru_dim]
        attn_in = attn_in.reshape(batch_size*seq_len, num_nodes, -1)
        
        attn_out, _ = self.attention(attn_in, attn_in, attn_in)
        attn_out = attn_out.reshape(batch_size, seq_len, num_nodes, -1)
        
        # Temporal aggregation
        aggregated = attn_out.mean(dim=1)  # [batch, nodes, 2*gru_dim]
        
        # Final prediction
        output = self.regressor(aggregated)  # [batch, nodes, 1]
        return output.squeeze(-1)  # [batch, nodes]

class EnhancedGRU(nn.Module):
    """Bidirectional GRU with Layer Normalization"""
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.norm = nn.LayerNorm(2*hidden_size)
        
    def forward(self, x):
        output, _ = self.gru(x)
        return self.norm(output)

# Test the architecture
if __name__ == "__main__":
    # Mock data
    batch_size = 4
    seq_len = 12
    num_nodes = 2139
    features = 3
    
    # Mock adjacency matrix
    adj = scipy.sparse.random(num_nodes, num_nodes, density=0.01)
    
    # Initialize model
    model = MSTSN_Gambia(adj_matrix=adj)
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, num_nodes, features)
    output = model(x)
    
    print("Architecture test successful!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should be [4, 2139]
