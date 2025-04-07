import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv  # More stable than GATConv

class AdaptiveAdjacency(nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.learned_adj = None

    def forward(self, batch_size):
        norm_embed = F.normalize(self.embedding, p=2, dim=1)
        adj = torch.mm(norm_embed, norm_embed.t())
        self.learned_adj = adj.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, nodes, nodes]
        return self.learned_adj

class BatchedGAT(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.gat = GATv2Conv(in_dim, out_dim, heads=heads, concat=True)
        self.norm = nn.LayerNorm(out_dim * heads)

    def forward(self, x, adj):
        # x: [batch, nodes, features]
        # adj: [batch, nodes, nodes]
        batch_size, num_nodes, _ = x.shape
        edge_indices = []
        
        # Convert batch adjacency matrices to edge_index format
        for b in range(batch_size):
            edge_index = adj[b].nonzero(as_tuple=False).t()
            edge_indices.append(edge_index)
        
        # Process each graph in batch sequentially (PyG limitation)
        outputs = []
        for b in range(batch_size):
            x_b = x[b]  # [nodes, features]
            edge_index = edge_indices[b]
            out = self.gat(x_b, edge_index)
            outputs.append(out)
        
        return self.norm(torch.stack(outputs))  # [batch, nodes, out_dim*heads]

class SpatialProcessor(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.adaptive_adj = AdaptiveAdjacency(num_nodes, hidden_dim)
        self.gat1 = BatchedGAT(in_dim, hidden_dim)
        self.gat2 = BatchedGAT(hidden_dim * 4, out_dim)  # heads=4
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: [batch, nodes, features]
        adj = self.adaptive_adj(x.size(0))  # [batch, nodes, nodes]
        x = F.relu(self.gat1(x, adj))
        x = self.dropout(x)
        x = self.gat2(x, adj)
        return x
