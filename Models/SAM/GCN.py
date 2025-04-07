import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class AdaptiveAdjacency(nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.learned_adj = None

    def forward(self):
        norm_embed = F.normalize(self.embedding, p=2, dim=1)
        self.learned_adj = torch.mm(norm_embed, norm_embed.t())
        return self.learned_adj

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim, heads=heads, concat=True)
        self.norm = nn.LayerNorm(out_dim * heads)

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        return self.norm(x)

class SpatialProcessor(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.adaptive_adj = AdaptiveAdjacency(num_nodes, hidden_dim)
        self.gat1 = GATLayer(in_dim, hidden_dim)
        self.gat2 = GATLayer(hidden_dim * 4, out_dim)  # Multiply by heads (4)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        adj = self.adaptive_adj()
        edge_index = adj.nonzero(as_tuple=False).t()  # Convert dense adj to edge_index
        x = F.relu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        return x
