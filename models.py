import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer
import scipy.sparse as sp
import numpy as np

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, adj, dataset):
        """Sparse version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        edge = adj._indices()

        m = sp.lil_matrix((adj.shape[0], edge.shape[1]))
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
            m[i,range(j, j + length[i])] = 1
            j += length[i]
            
        m = sparse_mx_to_torch_sparse_tensor(m)

        self.attentions = [GraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha,
                                                 adj=adj,
                                                 dataset=dataset,
                                                 edge=edge,
                                                 m=m,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha,
                                             adj=adj,
                                             dataset=dataset,
                                             edge=edge,
                                             m=m,
                                             concat=False)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x)
        return x

