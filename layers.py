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

    def __init__(self, in_features, out_features, dropout, alpha, adj, dataset, edge, concat=True, no_cuda=False):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.adj = adj
        self.dataset = dataset
        self.edge = edge
        
        cuda = not no_cuda and torch.cuda.is_available()

        if concat==False:
            in_features = 2*in_features
        self.W_high = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.W_low = nn.Parameter(torch.zeros(size=(in_features, out_features)))

        nn.init.xavier_normal_(self.W_high.data, gain=1.414)
        nn.init.xavier_normal_(self.W_low.data, gain=1.414)
                
        self.a_high = nn.Parameter(torch.zeros(size=(1, 1*out_features)))
        nn.init.xavier_normal_(self.a_high.data, gain=1.414)
        self.a_low = nn.Parameter(torch.zeros(size=(1, 1*out_features)))
        nn.init.xavier_normal_(self.a_low.data, gain=1.414)
        
        self.c_low = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.c_low.data, gain=1)
        self.c_high = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.c_high.data, gain=1)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def gam(self, x, epsilon=1e-6):
        return F.relu6(x+3)/3 + epsilon

    def relu_bt(self, x):
        threshold = torch.norm(x,p=float("inf")).clone().detach()
        return - torch.threshold(-F.leaky_relu(x),-threshold,-threshold)
    
    def forward(self, input):
        dv = 'cuda' if input.is_cuda else 'cpu'
        
        N = input.size()[0]
        edge = self.edge

      
        c_high = self.gam(self.c_high) - 1
        c_low = self.gam(self.c_low)

        h_high = torch.mm(input, self.W_high)
        h_low = torch.mm(input, self.W_low)
        h_high = self.relu_bt(h_high)
        h_low = self.relu_bt(h_low)
        #assert not torch.isnan(h_high).any()

        #input1 = torch.add((h_high[edge[0, :], :]), h_high[edge[1, :], :])
        #high_sub = torch.sub((h_high[edge[1, :], :]), h_high[edge[0, :], :])
        #input2 = torch.sub(h_low[edge[0, :], :], h_low[edge[1, :], :])
        #high_agg = torch.add((h_high[edge[0, :], :]), h_high[edge[1, :], :])
        #high_diff = torch.sub((h_high[edge[0, :], :]), h_high[edge[1, :], :])
        #low_agg = torch.add((h_low[edge[0, :], :]), h_low[edge[1, :], :])
        #low_diff = torch.sub((h_low[edge[0, :], :]), h_low[edge[1, :], :])
        edge_h_high = torch.sub((h_high[edge[0, :], :]), h_high[edge[1, :], :]).t()
        edge_h_low = torch.add((h_low[edge[0, :], :]), h_low[edge[1, :], :]).t()
        #h_add = self.relu_bt(h_add)
        #h_sub = self.relu_bt(h_sub)
        #low_agg = torch.add(h_low[edge[0, :], :], h_low[edge[1, :], :])
        
        #input1 = self.relu_bt(input1)
        #input2 = self.relu_bt(input2)
        #high_sub = self.relu_bt(high_sub)
        #low_agg = self.relu_bt(low_agg)
        #if not self.concat:
            # if this layer is last layer,
        #    input2 = torch.add((h_high[edge[0, :], :]), h_high[edge[1, :], :])
        
        # Self-attention on the nodes - Shared attention mechanism
        #edge_h_high = torch.cat((beta_high*h_high[edge[0, :], :], (2-beta_high)*h_high[edge[1, :], :], input1), dim=1).t()
        #edge_h_low = torch.cat((beta_low*h_low[edge[0, :], :], (2-beta_low)*h_low[edge[1, :], :], input2), dim=1).t()
        #edge_h_high = torch.add((h_high[edge[0, :], :], h_high[edge[1, :], :]), dim=1).t()
        #edge_h_low = torch.sub((h_low[edge[0, :], :], h_low[edge[1, :], :]), dim=1).t()
        #edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        #edge_h_high = torch.cat((h_high[edge[0, :], :], h_high[edge[1, :], :], high_agg, high_diff), dim=1).t()
        #edge_h_low = torch.cat((h_high[edge[0, :], :], h_high[edge[1, :], :], low_agg, low_diff), dim=1).t()
        #edge_h_high = torch.cat((h_high[edge[0, :], :], h_high[edge[1, :], :]), dim=1).t()
        #edge_h_low = torch.cat((h_low[edge[0, :], :], h_low[edge[1, :], :]), dim=1).t()

        edge_e_high = torch.exp(-self.leakyrelu(self.a_high.mm(edge_h_high).squeeze()))
        edge_e_low = torch.exp(-self.leakyrelu(self.a_low.mm(edge_h_low).squeeze()))
        assert not torch.isnan(edge_e_high).any()
        # edge_e: E

        edge_e_high = self.dropout(edge_e_high)
        edge_e_low = self.dropout(edge_e_low)
        assert not torch.isnan(edge_e_high).any()
        # edge_e: E

        e_high_rowsum = self.special_spmm(edge, edge_e_high, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        e_low_rowsum = self.special_spmm(edge, edge_e_low, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        assert not torch.isnan(e_high_rowsum).any()
        # e_rowsum: N x 1
            
        h_prime_high = self.special_spmm(edge, edge_e_high, torch.Size([N, N]), h_high)
        h_prime_low = self.special_spmm(edge, edge_e_low, torch.Size([N, N]), h_low)
        assert not torch.isnan(h_prime_high).any()
        # h_prime: N x out
        
        h_prime_high = h_prime_high.div(e_high_rowsum+1e-16)
        h_prime_low = h_prime_low.div(e_low_rowsum+1e-16)
        assert not torch.isnan(h_prime_high).any()
        # h_prime: N x out


        if self.concat:
            # if this layer is not last layer,
            h_prime = torch.cat((h_prime_high, h_prime_low), dim=1)
            assert not torch.isnan(h_prime).any()
            return self.relu_bt(h_prime)
        else:
            # if this layer is last layer,
            h_prime = 2*torch.add(c_high*h_prime_high, c_low*h_prime_low)/torch.abs(c_low+c_high)
            return self.relu_bt(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
