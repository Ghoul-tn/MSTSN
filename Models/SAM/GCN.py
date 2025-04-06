import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def cal_adj_norm(adj):
    node_num = adj.shape[0]
    adj = np.asarray(adj)
    adj_ = adj + np.eye(node_num)
    # 度矩阵
    d = np.sum(adj_,1)
    d_sqrt = np.power(d, -0.5)
    d_sqrt = np.diag(d_sqrt)
    adj_norm = np.dot(np.dot(d_sqrt, adj_), d_sqrt)
    return adj_norm

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = GraphConvolution(input_dim, hidden_dim)
        self.layer2 = GraphConvolution(hidden_dim, output_dim)
        
    def forward(self, x, adj):
        """Process batched graph data
        Args:
            x: (batch_size, num_nodes, input_dim)
            adj: (num_nodes, num_nodes)
        Returns:
            (batch_size, num_nodes, output_dim)
        """
        # Process each sample in batch
        batch_size = x.size(0)
        outputs = []
        adj = adj.to(x.device)
        
        for i in range(batch_size):
            x_i = x[i]  # (num_nodes, input_dim)
            x_i = self.layer1(x_i, adj)
            x_i = self.layer2(x_i, adj)
            outputs.append(x_i.unsqueeze(0))
            
        return torch.cat(outputs, dim=0)

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x, adj):
        # Linear transformation
        x = self.linear(x)  # (num_nodes, output_dim)
        
        # Graph propagation
        x = torch.mm(adj, x)  # (num_nodes, output_dim)
        
        return self.activation(x)
