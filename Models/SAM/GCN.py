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
    def __init__(self, input_dim, output_dim, activation='relu'):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.activation = nn.ReLU() if activation == 'relu' else nn.Sigmoid()

    def forward(self, x, adj):
        # x shape: (num_nodes, input_dim)
        # adj shape: (num_nodes, num_nodes)
        support = torch.mm(x, self.weight)  # (num_nodes, output_dim)
        output = torch.spmm(adj, support) + self.bias  # Sparse matrix multiplication
        return self.activation(output)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, adj):
        super().__init__()
        self.layer1 = GraphConvolution(input_dim, hidden_dim)
        self.layer2 = GraphConvolution(hidden_dim, output_dim, activation='sigmoid')
        self.adj = adj

    def forward(self, x):
        # x shape: (seq_len, batch_size, input_dim)
        seq_len, batch_size, _ = x.shape
        outputs = []
        
        for t in range(seq_len):
            x_t = x[t]  # (batch_size, input_dim)
            x_t = self.layer1(x_t, self.adj)
            x_t = self.layer2(x_t, self.adj)
            outputs.append(x_t)
            
        return torch.stack(outputs)  # (seq_len, batch_size, output_dim)
