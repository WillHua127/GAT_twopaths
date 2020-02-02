import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
    
    
class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class GraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, adj, concat=True, no_cuda=False):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.adj = adj
        
        edge = adj.nonzero().t()
        self.edge = edge
        m = torch.zeros(adj.shape[0], edge.shape[1])
        length = []
        count = 0
        for i in range(edge.shape[1]):
            if i == edge.shape[1]-1:
                length.append(count+1)
                break
            elif edge[0][i] == edge[0][i+1]:
                count = count+1
            else:
                count = count+1
                length.append(count)
                count = 0
        j = 0
        for i in range(len(length)):
            m[i, range(j, j + length[i])] = 1
            j += length[i]
            
        self.m = m
        cuda = not no_cuda and torch.cuda.is_available()
        if cuda:
            m = m.cuda()
            
        self.W_high = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W_high.data, gain=1.414)
        self.W_low = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W_low.data, gain=1.414)
                
        self.a_high = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a_high.data, gain=1.414)
        self.a_low = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a_low.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = self.edge

        h_high = torch.mm(input, self.W_high)
        h_low = torch.mm(input, self.W_low)
        # h: N x out
        
        # Self-attention on the nodes - Shared attention mechanism
        edge_h_high = torch.cat((h_high[edge[0, :], :], h_high[edge[1, :], :]), dim=1).t()
        edge_h_low = torch.cat((h_low[edge[0, :], :], h_low[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e_high = torch.exp(-self.leakyrelu(self.a_high.mm(edge_h_high).squeeze()))
        edge_e_low = torch.exp(-self.leakyrelu(self.a_low.mm(edge_h_low).squeeze()))
        # edge_e: E

        e_high_rowsum = self.special_spmm(edge, edge_e_high, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        e_low_rowsum = self.special_spmm(edge, edge_e_low, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e_high = F.relu6(edge_e_high)
        edge_e_high = self.dropout(edge_e_high)
        edge_e_low = F.relu6(edge_e_low)
        edge_e_low = self.dropout(edge_e_low)
        # edge_e: E

        h_high = torch.add(h_high[edge[0, :], :], h_high[edge[1, :], :])
        h_low = torch.sub(h_low[edge[0, :], :], h_low[edge[1, :], :])
        # E x D
        
        h_high = torch.matmul(self.m, h_high)
        h_low = torch.matmul(self.m, h_low)
            
        h_prime_high = self.special_spmm(edge, edge_e_high, torch.Size([N, N]), h_high)
        h_prime_low = self.special_spmm(edge, edge_e_low, torch.Size([N, N]), h_low)
        # h_prime: N x out
        
        h_prime_high = h_prime_high.div(e_high_rowsum)
        h_prime_low = h_prime_low.div(e_low_rowsum)
        # h_prime: N x out

        h_prime = 0.5 * torch.add(h_prime_high, h_prime_low)

        if self.concat:
            # if this layer is not last layer,
            return F.relu6(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

