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

class GCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, adj_matrix):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        
        # Register adjacency matrix as buffer
        adj_tensor = torch.FloatTensor(adj_matrix.toarray() if hasattr(adj_matrix, 'toarray') 
                                      else adj_matrix)
        self.register_buffer('adj', adj_tensor)

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_nodes, input_dim)
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Process each timestep
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # (batch_size, num_nodes, input_dim)
            
            # Graph convolution
            x_t = self.linear(x_t)  # (batch_size, num_nodes, output_dim)
            x_t = torch.bmm(self.adj.expand(batch_size, -1, -1), x_t)  # Graph propagation
            
            # Residual connection
            if x.shape[-1] == x_t.shape[-1]:  # Match dimensions
                x_t = x_t + x[:, t, :, :]
            
            x_t = self.norm(x_t)
            x_t = self.activation(x_t)
            outputs.append(x_t.unsqueeze(1))  # (batch_size, 1, num_nodes, output_dim)
        
        return torch.cat(outputs, dim=1)  # (batch_size, seq_len, num_nodes, output_dim)

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.activation = nn.ReLU()

    def forward(self, x, adj):
        # Ensure all tensors are on same device
        adj = adj.to(x.device)
        self.weight = self.weight.to(x.device)
        self.bias = self.bias.to(x.device)
        
        support = torch.mm(x, self.weight)  # (batch_size, output_dim)
        output = torch.mm(adj, support) + self.bias
        return self.activation(output)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = GraphConvolution(input_dim, hidden_dim)
        self.layer2 = GraphConvolution(hidden_dim, output_dim)

    def forward(self, x, adj):
        seq_len = x.size(1)
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_dim)
            x_t = self.layer1(x_t, adj)
            x_t = self.layer2(x_t, adj)
            outputs.append(x_t.unsqueeze(1))
            
        return torch.cat(outputs, dim=1)  # (batch_size, seq_len, output_dim)
