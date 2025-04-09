import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.utils.checkpoint import checkpoint
import torch_xla.core.xla_model as xm
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
        self.heads = heads
        self.per_head_dim = out_dim // heads
        self.gat = GATv2Conv(in_dim, self.per_head_dim, heads=heads, concat=True)
        
    def forward(self, x, adj):
        batch_size = x.size(0)
        # Remove explicit dtype casting
        outputs = torch.zeros(batch_size, x.size(1), self.per_head_dim * self.heads,
                            device=x.device)  # Let XLA decide dtype
        
        for b in range(batch_size):
            edge_index = adj[b].nonzero(as_tuple=False).t()
            outputs[b] = self.gat(x[b], edge_index)  # No float() conversion
            
        return outputs

class SpatialProcessor(nn.Module):
    def __init__(self, num_nodes, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.adaptive_adj = AdaptiveAdjacency(num_nodes, hidden_dim)
        self.gat1 = BatchedGAT(in_dim, hidden_dim)  # Output: hidden_dim
        self.gat2 = BatchedGAT(hidden_dim, out_dim)  # Output: out_dim (128)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Use functional dropout to save memory
        batch_size = x.size(0)
        
        # Calculate adjacency matrix but only for the current batch
        adj = self.adaptive_adj(batch_size)
        
        # Apply GAT layers with intermediate memory cleanup
        x = F.relu(self.gat1(x, adj))
        x = F.dropout(x, p=0.2, training=self.training)  # Functional dropout
        xm.mark_step()  # Tell XLA to materialize tensors here, reducing peak memory
        
        return self.gat2(x, adj)  # Output will be exactly out_dim
