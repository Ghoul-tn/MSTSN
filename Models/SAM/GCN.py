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
