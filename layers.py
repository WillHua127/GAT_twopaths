import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, no_cuda=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.no_cuda = no_cuda

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
       
        a_input_high = torch.cat([h_high.repeat(1, N_high).view(N_high * N_high, -1), h_high.repeat(N_high, 1)], dim=1).view(N_high, -1, 2 * self.out_features)
        a_input_low = torch.cat([h_high.repeat(1, N_low).view(N_low * N_low, -1), h_high.repeat(N_low, 1)], dim=1).view(N_low, -1, 2 * self.out_features)
        
        e_high = self.leakyrelu(torch.matmul(a_input_high, self.a_high).squeeze(2))
        e_low = self.leakyrelu(torch.matmul(a_input_low, self.a_low).squeeze(2))

        
        adj_unnormalized = torch.zeros_like(adj)
        one = torch.ones((1,1))
        if self.no_cuda:
            adj_unnormalized = adj_unnormalized.cuda()
            one = one.cuda()
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

