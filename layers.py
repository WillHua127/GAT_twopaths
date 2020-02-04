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

    def __init__(self, in_features, out_features, dropout, alpha, adj, dataset, edge, m, concat=True, no_cuda=False):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.adj = adj
        self.dataset = dataset
        self.edge = edge
        self.m = m
        
        cuda = not no_cuda and torch.cuda.is_available()
        if cuda:
            self.m = self.m.cuda()
            
        self.W_high = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W_high.data, gain=1.414)
        self.W_low = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W_low.data, gain=1.414)
                
        self.a_high = nn.Parameter(torch.zeros(size=(1, 4*out_features)))
        nn.init.xavier_normal_(self.a_high.data, gain=1.414)
        self.a_low = nn.Parameter(torch.zeros(size=(1, 4*out_features)))
        nn.init.xavier_normal_(self.a_low.data, gain=1.414)
        
        #self.c_high = nn.Parameter(torch.zeros(size=(1, 1)))
        #nn.init.xavier_uniform_(self.c_high.data, gain=1)
        #self.c_low = nn.Parameter(torch.zeros(size=(1, 1)))
        #nn.init.xavier_uniform_(self.c_low.data, gain=1)

        self.g_high = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.g_high.data, gain=1)
        self.b_high = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.b_high.data, gain=1)
        self.g_low = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.g_low.data, gain=1)
        self.b_low = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.b_low.data, gain=1)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def gam(self, x, epsilon=1e-6):
        return F.relu6(x+3)/3 + epsilon

    def relu_bt(self, x):
        threshold = torch.norm(x,p=float("inf")).clone().detach()#(x.shape[0] * x.shape[1])#torch.Tensor([1e-6]).cuda()#torch.norm(x,p=float("inf")).clone().detach()#/(x.shape[0] * x.shape[1]) # p=1,2,float("inf"), torch.Tensor([1e-6]).cuda()
        #print(" threshold: ", threshold)
        return - torch.threshold(-F.leaky_relu(x),-threshold,-threshold) #F.relu(x-threshold) #- torch.threshold(-F.relu(x),-threshold,-threshold)
        #return F.relu6(x)
        #return F.relu(x-threshold)+threshold
        #return torch.exp(F.relu(x-threshold)+threshold)-torch.Tensor([1])
        #return torch.tanh(F.relu(x-threshold)) #
        #return torch.nn.functional.gelu(x)
        #return F.elu(x)
        #return F.relu(x)
    
    def forward(self, input):
        dv = 'cuda' if input.is_cuda else 'cpu'

        if self.dataset == 'citeseer':
            N = self.adj.size()[0]
        else:
            N = input.size()[0]
        edge = self.edge

        gamma_high = self.gam(self.g_high)
        beta_high = self.gam(self.b_high)
        gamma_low = self.gam(self.g_low)
        beta_low = self.gam(self.b_low)
        #c_high = self.gam(self.c_high)
        #c_low = self.gam(self.c_low) - 1

        h_high = torch.mm(input, self.W_high)
        h_low = torch.mm(input, self.W_low)
        h_high = self.relu_bt(h_high)
        h_low = self.relu_bt(h_low)
        # h: N x out
        assert not torch.isnan(h_high).any()

        input1 = torch.sub((h_high[edge[0, :], :]), h_high[edge[1, :], :])
        input2 = torch.add(h_low[edge[0, :], :], h_low[edge[1, :], :])
        
        # Self-attention on the nodes - Shared attention mechanism
        edge_h_high = torch.cat((beta_high*h_high[edge[0, :], :], (2-beta_high)*h_high[edge[1, :], :], gamma_high*input1, (2-gamma_high)*input2), dim=1).t()
        edge_h_low = torch.cat((beta_low*h_low[edge[0, :], :], (2-beta_low)*h_low[edge[1, :], :], gamma_low*input1, (2-gamma_low)*input2), dim=1).t()
        assert not torch.isnan(edge_h_high).any()
        # edge: 2*D x E

        edge_e_high = torch.exp(-self.leakyrelu(torch.div(self.a_high.mm(edge_h_high).squeeze(), torch.norm(self.a_high))))
        edge_e_low = torch.exp(-self.leakyrelu(torch.div(self.a_low.mm(edge_h_low).squeeze(), torch.norm(self.a_low))))
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

        
        h_high = torch.matmul(self.m, input1)
        h_low = torch.matmul(self.m, input2)
        assert not torch.isnan(h_high).any()
            
        h_prime_high = self.special_spmm(edge, edge_e_high, torch.Size([N, N]), h_high)
        h_prime_low = self.special_spmm(edge, edge_e_low, torch.Size([N, N]), h_low)
        assert not torch.isnan(h_prime_high).any()
        # h_prime: N x out
        
        h_prime_high = h_prime_high.div(e_high_rowsum+1e-16)
        h_prime_low = h_prime_low.div(e_low_rowsum+1e-16)
        assert not torch.isnan(h_prime_high).any()
        # h_prime: N x out

        #h_prime_high = c_high * h_prime_high
        #h_prime_low = c_low * h_prime_low

        h_prime = torch.add(h_prime_high, h_prime_low)
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return self.relu_bt(h_prime)
        else:
            # if this layer is last layer,
            return self.relu_bt(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
