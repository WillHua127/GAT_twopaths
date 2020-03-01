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

    
    def __init__(self, in_features, out_features, dropout, alpha, adj, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.adj = adj
        
        edge = adj._indices()
        self.edge = edge

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        #self.a2 = nn.Parameter(torch.zeros(size=(1, 1*out_features)))
        #nn.init.xavier_normal_(self.a2.data, gain=1.414)
        #self.a3 = nn.Parameter(torch.zeros(size=(1, 1*out_features)))
        #nn.init.xavier_normal_(self.a3.data, gain=1.414)
        #self.WT = nn.Parameter(torch.zeros(size=(4*out_features, 4*out_features)))
        #nn.init.xavier_normal_(self.WT.data, gain=1.414)
        
        #self.g = nn.Parameter(torch.zeros(size=(1, 1)))
        #nn.init.xavier_uniform_(self.g.data, gain=1)
        
        #self.b = nn.Parameter(torch.zeros(size=(1, 1)))
        #nn.init.xavier_uniform_(self.b.data, gain=1)
        
        #self.bias = nn.Parameter(torch.zeros(size=(adj.shape[0], out_features)))
        #nn.init.xavier_uniform_(self.bias.data, gain=1)
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

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
    
    def gam(self, x, epsilon=1e-6):
        return F.relu6(x+3)/3 + epsilon
    
    
    def forward(self, input):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = self.edge
        #gamma = self.gam(self.g)
        #beta = self.gam(self.b)
        

        h = torch.mm(input, self.W)
        #h = self.relu_bt(h)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        agg = self.relu_bt(torch.add(h[edge[0, :], :], h[edge[1, :], :]))       
        diff = self.relu_bt(torch.sub(h[edge[0, :], :], h[edge[1, :], :]))
        #if not self.concat:
        #    input2 = torch.add(h[edge[0, :], :], h[edge[1, :], :])
        edge_h = torch.cat([h[edge[0, :], :], h[edge[1, :], :]], dim=1).t()
        #edge_h = torch.mm(self.WT, edge_h)
        # edge: 2*D x E
        #edge_h = torch.cat([h[edge[0, :], :], h[edge[1, :], :]], dim=1).t()

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        #edge_e_2 = torch.exp(-self.leakyrelu(self.a2.mm(edge_h_2).squeeze()))
        #edge_e_3 = torch.exp(-self.leakyrelu(self.a3.mm(edge_h_3).squeeze()))
        
        #assert not torch.isnan(edge_e).any()
        # edge_e: E
        
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        #e_rowsum_2 = self.special_spmm(edge, edge_e_2, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        #e_rowsum_3 = self.special_spmm(edge, edge_e_3, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        #edge_e = edge_e_1+edge_e_2+edge_e_3
        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        #assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        #e_rowsum = e_rowsum_1+e_rowsum_2+e_rowsum_3
        h_prime = h_prime.div(e_rowsum+1e-16)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return self.relu_bt(h_prime)
        else:
            # if this layer is last layer,
            return self.relu_bt(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
