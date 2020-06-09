import numpy as np
import networkx as nx
from scipy.sparse import diags, identity
from copy import deepcopy
import random
import math
from munkres import Munkres, print_matrix
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear
from sklearn import metrics

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.utils import add_remaining_self_loops, to_undirected

def generate_bigram(ls):
    res = []
    for i in range(len(ls)-1):
        res += [(ls[i], item) for item in ls[i+1:]]
    return res

def max_lists(lists):
    return max([item for items in lists for item in items])
def min_lists(lists):
    return min([item for items in lists for item in items])

def max_node(G):
    return max(G)
def min_node(G):
    return min(G)

def top_k_prediction(preds, labels):
    preds = preds.data.cpu().numpy()
    for idx, item in enumerate(labels):
        preds_tmp = np.zeros((121,))
        preds_tmp[preds[idx].argsort()[-sum(item):]] = 1
        preds[idx] = preds_tmp
    return preds

def fill_diagonal(A, value, device):
    _A = A.clone()
    mask = torch.eye(A.shape[0], A.shape[1]).bool().to(device)
    _A = _A.masked_fill_(mask, value)
    return _A

# def pairwise_distances(x, y=None):
#     '''
#     Input: x is a Nxd matrix
#            y is an optional Mxd matirx
#     Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
#             if y is not given then use 'y=x'.
#     i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
#     '''
#     x_norm = (x**2).sum(1).view(-1, 1)
#     if y is not None:
#         y_norm = (y**2).sum(1).view(1, -1)
#     else:
#         y = x
#         y_norm = x_norm.view(1, -1)

#     dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
#     return dist
def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT 
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def non_zero_mean(A):
    res = A.sum() / torch.nonzero(A).size(0)
    return res.data

def sparse_softmax(M, dim):
    """
    input:
    M: 2-D tensor
    dim: int, 0, 1 or -1
    output:
    out: 2-D tensor with same size as M
    """
    # exp of all elements
    M_exp = M.exp()
    # replace the exp of 0 by 0 (approxime 1e-16 to avoid NAN)
    zeros = torch.full(M_exp.size(), 1e-16).to(M_exp.device)
    M_exp = torch.where(M_exp==1, zeros, M_exp)
    Sum = M_exp.sum(dim)
    # soft max operation
    out = M_exp / Sum
    # out = M_exp.clone() / Sum.clone()

    return out


# # not usable for GC
# def attention(query, message, lin_att, normalize, drop_ratio, training):
#     if normalize:
#         query = F.normalize(query, p=2, dim=-1)
#         message = F.normalize(message, p=2, dim=-1)

#     # calculate attention score
#     score = lin_att(torch.cat((message, query), dim=-1)).squeeze(-1)
#     score = F.leaky_relu(score)

#     score = F.softmax(score)
    
#     # Sample attention coefficients stochastically.
#     score = F.dropout(score, p=drop_ratio, training=training)

#     return score.unsqueeze(-1)

# usable for GC
def attention(query, message, lin_att, normalize, drop_ratio, training):
    if normalize:
        query = F.normalize(query, p=2, dim=-1)
        message = F.normalize(message, p=2, dim=-1)

    # calculate attention score
    score = lin_att(torch.cat((message, query), dim=-1))
    score = F.leaky_relu(score)

    score = F.softmax(score)
    
    # Sample attention coefficients stochastically.
    score = F.dropout(score, p=drop_ratio, training=training)

    return score


def sparse_attention(query, message, query_M, lin_att, normalize, drop_ratio, training):
    if normalize:
        query = F.normalize(query, p=2, dim=-1)
        message = F.normalize(message, p=2, dim=-1)

    # calculate attention score
    score = lin_att(torch.cat((message, query), dim=-1))
    score = F.leaky_relu(score)
    score = query_M * score
    # Sample attention coefficients stochastically.
    score = F.dropout(score, p=drop_ratio, training=training)
    score = sparse_softmax(M=score, dim=0)
    score = score.sum(-1).view(-1, 1)
    
    return score


def gating_sum(orig, received, lin_1, lin_2, normalize, drop_ratio, training):
    if normalize:
        orig = F.normalize(orig, p=2, dim=-1)
        received = F.normalize(received, p=2, dim=-1)
    
    gat_score = F.sigmoid(lin_1(orig) + lin_2(received))
    gat_score = F.dropout(gat_score, p=drop_ratio, training=training)
    out = torch.mul(gat_score, orig) + torch.mul((1-gat_score), orig)
    
    return out


def smooth_filter(laplacian_matrix, lda=0.1):
    dim = laplacian_matrix.shape[0]
    degree_matrix_vec = laplacian_matrix.diagonal()
    #self_loop_vec = degree_matrix_vec * lda
    degree_matrix = diags(degree_matrix_vec, 0)
    #self_loop = diags(self_loop_vec, 0)
    adj_matrix = degree_matrix - laplacian_matrix + lda * identity(dim)
    #adj_matrix = degree_matrix - laplacian_matrix + self_loop
    degree_vec = adj_matrix.sum(axis=1)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.squeeze(np.asarray(np.power(degree_vec, -0.5)))
    d_inv_sqrt[np.isinf(d_inv_sqrt)|np.isnan(d_inv_sqrt)] = 0
    degree = diags(d_inv_sqrt, 0)
    norm_adj = degree @ adj_matrix @ degree
    return norm_adj

def adj2norm_adj(matrix):
    device = matrix.device

    matrix = matrix.data.cpu().numpy()
    G = nx.from_numpy_array(matrix)
    laplacian = nx.linalg.laplacianmatrix.laplacian_matrix(G, nodelist=None, weight='weight')
    norm_adj = torch.FloatTensor(smooth_filter(laplacian_matrix=laplacian).toarray()).to(device)

    return norm_adj

def generate_top_down_graph(embeddings):
    num_nodes = embeddings[0].shape[0]
    device = embeddings[0].device

    idx_1, idx_2 = [], []
    for i in range(num_nodes):
        for j in range(len(embeddings)-1):
            idx_1 += [i+num_nodes*(j+1)]
            idx_2 += [i]
            # idx_1 += [i, i+num_nodes*(j+1)]
            # idx_2 += [i+num_nodes*(j+1), i]
        # idx_1 += [i]
        # idx_2 += [i]
            
    edge_index = torch.Tensor([idx_1, idx_2]).long().to(device)
    # edge_index = torch.Tensor([idx_2]).long().to(device)
    # edge_index, _ = add_remaining_self_loops(edge_index=edge_index, edge_weight=None)
    
    return edge_index

# def generate_top_down_graph(embeddings):
#     num_nodes = embeddings[0].shape[0]
#     device = embeddings[0].device
    
#     idx_1, idx_2 = [], []
#     for i in range(num_nodes):
#         candidates = [i]
#         for j in range(len(embeddings)-1):
#              candidates += [i+num_nodes*(j+1)]
#         pairs = generate_bigram(candidates)
#         # idx_1 += [item[0] for item in pairs]
#         # idx_2 += [item[1] for item in pairs]
#         idx_1 += [item[1] for item in pairs]
#         idx_2 += [item[0] for item in pairs]
    
#     edge_index = torch.Tensor([idx_1, idx_2]).long().to(device)
#     edge_index, _ = add_remaining_self_loops(edge_index=edge_index, edge_weight=None)
    
#     return edge_index


def recon_loss(z, pos_edge_index):
    r"""Given latent variables :obj:`z`, computes the binary cross
    entropy loss for positive edges :obj:`pos_edge_index` and negative
    sampled edges.

    Args:
        z (Tensor): The latent space :math:`\mathbf{Z}`.
        pos_edge_index (LongTensor): The positive edges to train against.
    """
    EPS = 1e-15
    decoder = tg.nn.InnerProductDecoder()

    pos_loss = -torch.log(
        decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

    # Do not include self-loops in negative samples
    pos_edge_index, _ = tg.utils.remove_self_loops(pos_edge_index)
    pos_edge_index, _ = tg.utils.add_self_loops(pos_edge_index)

    neg_edge_index = tg.utils.negative_sampling(pos_edge_index, z.size(0))
    neg_loss = -torch.log(1 -
                            decoder(z, neg_edge_index, sigmoid=True) +
                            EPS).mean()

    return pos_loss + neg_loss


def kl_loss(mu, logvar):
    r"""Computes the KL loss, for the passed arguments :obj:`mu`
    and :obj:`logvar`.

    Args:
        mu (Tensor, optional): The latent space for :math:`\mu`.
        logvar (Tensor, optional): The latent space for
            :math:`\log\sigma^2`.
    """
    
    loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1))
    loss = (1 / mu.shape[0]) * loss

    return loss

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_test_split_edges(data, val_ratio, test_ratio, SEED):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.

    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)

    :rtype: :class:`torch_geometric.data.Data`
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    assert 'batch' not in data  # No batch-mode.

    row, col = data.edge_index
    data.edge_index = None

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    # Negative edges.
    num_nodes = data.num_nodes
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero().t()
    perm = random.sample(range(neg_row.size(0)),
                         min(n_v + n_t, neg_row.size(0)))
    perm = torch.tensor(perm)
    perm = perm.to(torch.long)
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data


def cluster_acc(y_true, y_pred):
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print('error')
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    precision_macro = metrics.precision_score(y_true, new_predict, average='macro')
    recall_macro = metrics.recall_score(y_true, new_predict, average='macro')
    f1_micro = metrics.f1_score(y_true, new_predict, average='micro')
    precision_micro = metrics.precision_score(y_true, new_predict, average='micro')
    recall_micro = metrics.recall_score(y_true, new_predict, average='micro')
    return acc, f1_macro


def eva(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    print(epoch, ':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
            ', f1 {:.4f}'.format(f1))
    print('\n')

