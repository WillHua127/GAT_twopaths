import numpy as np
import random
import json
import sys
import os
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
import math
import torch 

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_dataset(dataset, dense=False):
    dense_adj, features, labels = torch.load("%s_dense_adj.pt" % dataset), torch.load("%s_features.pt" % dataset), torch.load("%s_labels.pt" % dataset)

    y=140
    s_val = labels[np.random.choice(labels.shape[0], 500, replace=False)]
    if dataset == 'citeseer':
        y=120
    elif dataset == 'pubmed':
        y=60
    idx_train = range(y)
    idx_val = range(y, y + 500)
    idx_test = range(y + 500, y + 1500)

    if dataset == 'cora':
        idx_train, idx_val, idx_test = range(140), range(140, 640), range(1708, 2708)
    elif dataset == 'citeseer':
        idx_train, idx_val, idx_test = range(120), range(120, 620), range(2312, 3312)
    elif dataset == 'pubmed':
        idx_train, idx_val, idx_test = range(60), range(60, 560), range(18717, 19717)
        
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    if dense:
        return dense_adj, features, labels, idx_train, idx_val, idx_test
    else:
        indices = torch.nonzero(dense_adj).t(); values = dense_adj[indices[0], indices[1]]
        adj = torch.sparse.FloatTensor(indices, values, dense_adj.size()).clone()
        del dense_adj, indices, values
        return adj, features, labels, idx_train, idx_val, idx_test
    



def load_data(prefix):
    print('Loading {} dataset...'.format(prefix))
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("./dataset/ind.{}.{}".format(prefix, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
        
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./dataset/ind.{}.test.index".format(prefix))   
    test_idx_range = np.sort(test_idx_reorder)

    if prefix == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder),
                                    max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
    
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)
    
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = sp.csr_matrix(adj, dtype=np.float32)[:,list(np.where(np.sum(labels,1)==1)[0])][list(np.where(np.sum(labels,1)==1)[0]),:]
    adj = sp.coo_matrix(adj,dtype=np.float32)
    adj = normalize(adj + sp.eye(adj.shape[0])) 

    # build symmetric adjacency matrix
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    return adj, features, labels, 



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

