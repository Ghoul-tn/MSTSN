import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.utils.checkpoint import checkpoint

class AdaptiveAdjacency(nn.Module):
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.learned_adj = None

    def forward(self, batch_size):
        norm_embed = F.normalize(self.embedding, p=2, dim=1)
        adj = torch.mm(norm_embed, norm_embed.t())
        return adj.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, nodes, nodes]

class BatchedGAT(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.gat = GATv2Conv(in_dim, out_dim // heads, heads=heads, concat=True)  # Ensure output is out_dim
        self.norm = nn.LayerNorm(out_dim)
        
    def _forward(self, x, adj):
        batch_size, num_nodes, _ = x.shape
        outputs = []
        for b in range(batch_size):
            edge_index = adj[b].nonzero(as_tuple=False).t()
            out = self.gat(x[b], edge_index)
            outputs.append(out)
        return self.norm(torch.stack(outputs))

class SpatialProcessor(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.adaptive_adj = AdaptiveAdjacency(num_nodes, hidden_dim)
        self.gat1 = BatchedGAT(in_dim, hidden_dim)  # Output: hidden_dim
        self.gat2 = BatchedGAT(hidden_dim, out_dim)  # Output: out_dim (128)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        adj = self.adaptive_adj(x.size(0))
        x = F.relu(self.gat1(x, adj))
        x = self.dropout(x)
        return self.gat2(x, adj)  # Output will be exactly out_dim
