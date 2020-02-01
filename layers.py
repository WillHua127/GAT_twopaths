import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W_high = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_high.data, gain=1.414)
        self.W_low = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_low.data, gain=1.414)
        self.a_high = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a_high.data, gain=1.414)
        self.a_low = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a_low.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h_high = torch.mm(input, self.W_high)
        h_low = torch.mm(input, self.W_low)
        h_high = F.relu6(h_high)
        h_low = F.relu6(h_low)
        
        N_high = h_high.size()[0]
        N_low = h_low.size()[0]
        #addition
        #input1 = torch.add(h.repeat(1, N).view(N * N, -1), h.repeat(N, 1))
        #subtraction
        #input2 = torch.sub(h.repeat(1, N).view(N * N, -1), h.repeat(N, 1))
        #old
        a_input_high = torch.cat([h_high.repeat(1, N_high).view(N_high * N_high, -1), h_high.repeat(N_high, 1)], dim=1).view(N_high, -1, 2 * self.out_features)
        a_input_low = torch.cat([h_high.repeat(1, N_low).view(N_low * N_low, -1), h_high.repeat(N_low, 1)], dim=1).view(N_low, -1, 2 * self.out_features)
        #concat
        #a_input = torch.cat([h.repeat(1, N).view(N * N, -1), input1, input2], dim=1).view(N, -1, 3 * self.out_features)
        e_high = self.leakyrelu(torch.matmul(a_input_high, self.a_high).squeeze(2))
        e_low = self.leakyrelu(torch.matmul(a_input_low, self.a_low).squeeze(2))

        #zero_vec_high = -9e15*torch.ones_like(e_high)
        #zero_vec_low = -9e15*torch.ones_like(e_low)
        #attention_high = torch.where(adj > 0, e_high, zero_vec_high)
        #attention_low = torch.where(adj > 0, e_low, zero_vec_low)
        adj_unnormalized = torch.zeros_like(adj)
        one = torch.ones((1,1))
        adj_unnormalized = torch.where(adj > 0, one, adj_unnormalized)
        attention_high = torch.matmul(e_high, adj_unnormalized)
        attention_low = torch.matmul(e_low, adj_unnormalized)
        attention_high = F.relu6(attention_high)
        attention_high = F.softmax(attention_high, dim=1)
        attention_low = F.relu6(attention_low)
        attention_low = F.softmax(attention_low, dim=1)
        
        attention_high = F.dropout(attention_high, self.dropout, training=self.training)
        attention_low = F.dropout(attention_low, self.dropout, training=self.training)
        
        h_high = torch.add(h_high.repeat(1, N_high).view(N_high * N_high, -1), h_high.repeat(N_high, 1)).view(N_high, -1, self.out_features)
        h_high = torch.sum(h_high, dim=1)
        h_low = torch.sub(h_low.repeat(1, N_low).view(N_low * N_low, -1), h_low.repeat(N_low, 1)).view(N_low, -1, self.out_features) 
        h_low = torch.sum(h_low, dim=1)
        
        h_high = torch.matmul(attention_high, h_high)
        h_low = torch.matmul(attention_low, h_low)
        h_prime = 0.5 * torch.add(h_high, h_low)
        h_prime = F.relu6(h_prime)
        
        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


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

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
