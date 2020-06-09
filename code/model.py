import pandas as pd
import numpy as np
import random
import networkx as nx
import time
from copy import deepcopy
from sys import platform
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, normalized_mutual_info_score
from load_dataset import *
from set_up_training import *
from utils import *
import sys

import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import torch_geometric as tg
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, JumpingKnowledge
from torch_geometric.nn.pool.topk_pool import topk

import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline


def Generate_high_order_adjacency_matrix(A, order):
    As = [A]
    for _ in range(order-1):
        _A = torch.mm(A, A)
        _A = fill_diagonal(A=_A, value=0, device=_A.device)
        _A = (_A > 0.5).float() * 1
        for a in As:
            _A = _A - a
        A = _A
        A[A < 0] = 0
        As.append(A)
    return A

def Generate_clusters(matrix, edge_score, cluster_range, threshold):
    matrix = fill_diagonal(A=matrix, value=0, device=matrix.device)
    matrix[torch.where(edge_score < threshold)] = 0
    
    cluster_matrices = []
    for order in range(1, cluster_range+1):
        cluster_matrices.append(Generate_high_order_adjacency_matrix(A=matrix, order=order))
    
    return cluster_matrices

class Cluster_assignment(nn.Module):
    def __init__(self, cluster_range, device):
        super(Cluster_assignment, self).__init__()
        self.cluster_range = cluster_range
        self.device = device
        
    def forward(self, fitness, cluster_matrices):
        # input: fitness, cluster_matrix, cluster_matrices
        # output: cluster fitness
        cluster_fitness = []
        for matrix in cluster_matrices:
            # make sure scores are positive
            matrix_fitness = torch.mul(fitness, matrix)
            scores = matrix_fitness.sum(-1) / matrix.sum(-1) # cluster mean scores without weight
            scores[scores != scores] = 0 # we need it: 
#             scores[scores != scores] = 1 # we need it: 
            cluster_fitness.append(scores)
        
        cluster_scores = torch.stack(cluster_fitness).sum(0) / self.cluster_range # n*1
        
#         # set isolated node's score as 0
#         isolated_nodes = [cluster_matrix.sum(dim=1) == 0] # ???
#         cluster_scores[isolated_nodes[0]] = 0 # ???
        
        return cluster_scores

def Select_clusters(config, fitness, edge_matrix, cluster_matrix, cluster_matrices, cluster_scores, pooling_ratio, cluster_range, pick_all, overlap):
    # input: cluster_matrix, cluster_scores, pooling_ratio
    # output: cluster_fitness
    num_nodes = cluster_matrix.shape[0]
    if overlap:
        _cluster_matrix = cluster_matrices[0]
        local_extrema = (cluster_scores.expand(num_nodes, num_nodes)).t()\
                            - torch.mul(_cluster_matrix, cluster_scores.expand(num_nodes, num_nodes)) > 0
    else:
        cluster_matrices = Generate_clusters(matrix=edge_matrix,
                                             edge_score=fitness, 
                                             cluster_range=cluster_range*2,
                                             threshold=config['edge_threshold'])
        # represent clusters within one matrix 
        _cluster_matrix = torch.stack(cluster_matrices).sum(0)
        _cluster_matrix = (_cluster_matrix > 0.5).float().to(_cluster_matrix.device)
        local_extrema = (cluster_scores.expand(num_nodes, num_nodes)).t()\
                            - torch.mul(_cluster_matrix, cluster_scores.expand(num_nodes, num_nodes)) > 0
    
    if pick_all:
        # pick up all clusters
        cluster_ids = torch.tensor(torch.nonzero(local_extrema.sum(-1)==num_nodes).view(-1))
    else:
        # select top-k clusters
        candicate_clusters = torch.tensor(torch.nonzero(local_extrema.sum(-1)==num_nodes).view(-1))
        if candicate_clusters.shape[0] < num_nodes*pooling_ratio:
            cluster_ids = candicate_clusters
        else:
            candicate_cluster_fitness = cluster_scores[candicate_clusters]
            idx_ = topk(x=candicate_cluster_fitness, 
                        ratio=(num_nodes*pooling_ratio/candicate_clusters.shape[0]), 
                        batch=torch.tensor([0]*candicate_clusters.shape[0]).to(candicate_cluster_fitness.device)) # 
            cluster_ids = candicate_clusters[idx_]
    
    return cluster_ids

def Generate_assignment_matrix(cluster_matrix, cluster_ids, fitness, do_view):
    # input: cluster_matrix, cluster_idx
    # output: S, B
    num_nodes = cluster_matrix.shape[0]
    S = cluster_matrix.clone()
    S_fitness = torch.mul(cluster_matrix, fitness)
    
    reduced_nodes = torch.nonzero(S[:, cluster_ids].sum(-1)>0).view(-1).to(cluster_ids.device)
    # remove isolated nodes
    isolated_nodes = torch.nonzero(S.sum(0)==0).view(-1).to(cluster_ids.device)
    reduced_nodes = torch.cat([isolated_nodes, reduced_nodes])
    keeping_nodes = torch.tensor(list(set(range(num_nodes))-\
                                      set(cluster_ids.tolist())-\
                                      set(reduced_nodes.tolist()))).to(cluster_ids.device)
    
    if do_view:
        print('total: {}, next: {}, cluster: {}, isolated: {}, reduced: {}, keeping: {}.'.format(num_nodes, 
                                                                                                 len(cluster_ids)+len(keeping_nodes),
                                                                                                 len(cluster_ids),
                                                                                                 len(isolated_nodes),
                                                                                                 len(reduced_nodes),
                                                                                                 len(keeping_nodes)))
    
    if keeping_nodes.shape[0]>0:
        S[:, keeping_nodes]=0
        S_fitness[:, keeping_nodes]=0
    
    _S_ = torch.eye(n=S.shape[0], m=S.shape[1]).to(S.device)
    _S_ = _S_[:, list(set(range(num_nodes))-set(reduced_nodes.tolist()))]
    
    S_orig = S.clone()
    S = fill_diagonal(A=S, value=1, device=S.device)
    S = S[:, list(set(range(num_nodes))-set(reduced_nodes.tolist()))]
    
    S_fitness_orig = S_fitness.clone()
    S_fitness = fill_diagonal(A=S_fitness, value=1, device=S_fitness.device)
    S_fitness = S_fitness[:, list(set(range(num_nodes))-set(reduced_nodes.tolist()))]
    
    return S, S_orig, S_fitness, S_fitness_orig, keeping_nodes, _S_

class Merge_xs(nn.Module):
    def __init__(self, config, mode, dim, num_levels):
        super(Merge_xs, self).__init__()
        self.config=config
        self.mode=mode
        self.dim=dim
        self.num_levels=num_levels
        
        if self.mode=='LINEAR':
            self.out_cat = JumpingKnowledge(mode='cat', channels=self.dim, num_layers=self.num_levels)
            self.lin_top_down = nn.Linear(self.dim*self.num_levels, self.dim)
        elif self.mode=='MAX':
            self.out_cat = JumpingKnowledge(mode='max', channels=self.dim, num_layers=self.num_levels)
        elif self.mode=='LSTM':
            self.out_cat = JumpingKnowledge(mode='lstm', channels=self.dim, num_layers=self.num_levels)
        elif self.mode=='GCN':
            self.out_cat = GCNConv(self.dim, self.dim)
            self.out_cat_2 = GCNConv(self.dim, self.dim)
        elif self.mode=='GAT':
            self.out_cat = GATConv(self.dim, 8, heads=8)
            self.out_cat_2 = GATConv(8*8, self.dim, heads=1)
        
        self.lin_1 = nn.Linear(self.dim, self.dim)
        self.lin_2 = nn.Linear(self.dim, self.dim)
        if self.mode=='ATT':
            self.gat_atts = nn.ModuleList()
            for _ in range(self.num_levels-1):
                self.gat_atts.append(nn.Linear(2*self.dim, 1))
            
    def forward(self, data, xs):
        generated_embeddings = xs
        scores = []
        
        if self.mode=='NONE':
            embedding = generated_embeddings[0]
        elif self.mode=='MEAN':
            # mean
            embedding = torch.mean(torch.stack(generated_embeddings), dim=0)
        elif self.mode=='GAT':
            all_embeddings = torch.cat(generated_embeddings, dim=0)
            # all_embeddings[0:generated_embeddings[0].shape[0],:] = 0
            top_down_edge_index = generate_top_down_graph(embeddings=generated_embeddings)
            # entire final aggregation
            edge_index = data.edge_index # with within level
            edge_index = torch.cat((edge_index, top_down_edge_index), dim=-1)
            # edge_index = top_down_edge_index # without within level
            # edge_index, _ = tg.utils.remove_self_loops(edge_index=edge_index, edge_attr=None)
            # edge_index, _ = tg.utils.add_remaining_self_loops(edge_index=edge_index, edge_weight=None)
            all_embeddings = self.out_cat(x=all_embeddings, edge_index=edge_index)
            all_embeddings = F.relu(all_embeddings)
            # all_embeddings = F.dropout(all_embeddings, p=self.config['drop_ratio'], training=self.training)
            all_embeddings = self.out_cat_2(x=all_embeddings, edge_index=edge_index)
            # all_embeddings = F.relu(all_embeddings)
            
            # embedding = all_embeddings[0:generated_embeddings[0].shape[0],:]
            embedding = gating_sum(orig=xs[0], 
                                    received=all_embeddings[0:xs[0].shape[0], :], 
                                    lin_1=self.lin_1, 
                                    lin_2=self.lin_2, 
                                    normalize=True,
                                    drop_ratio=self.config['drop_ratio'], 
                                    training=self.training)
        
        elif self.mode=='LINEAR':
            # linear transfer
            embedding = self.out_cat(xs=generated_embeddings)
            embedding = self.lin_top_down(embedding)
        
        elif self.mode=='ATT':
            query = generated_embeddings[0]
            messages = generated_embeddings[1:]
            for idx, m in enumerate(messages):
                score = attention(query=query,
                                    message=m,
                                    lin_att=self.gat_atts[idx],
                                    normalize=True,
                                    drop_ratio=self.config['drop_ratio'],
                                    training=self.training)
                messages[idx] = m * score
                scores.append(score)
                
            embedding = torch.sum(torch.stack([query]+messages), dim=0)
        
        elif self.mode=='GATING':
            query = generated_embeddings[0]
            messages = generated_embeddings[1:]
            for idx, m in enumerate(messages):
                score = attention(query=query,
                                    message=m,
                                    lin_att=self.gat_atts[idx],
                                    normalize=True,
                                    drop_ratio=self.config['drop_ratio'],
                                    training=self.training)
                messages[idx] = m * score
            
            # or above will get overfitting? train accuracy --> 1.0
            message = torch.sum(torch.stack(messages), dim=0)
            message = torch.mean(torch.stack(messages), dim=0)
            embedding = gating_sum(orig=query, 
                                    received=message, 
                                    lin_1=self.lin_1, 
                                    lin_2=self.lin_2, 
                                    normalize=True,
                                    drop_ratio=self.config['drop_ratio'], 
                                    training=self.training)

        else:
            embedding = self.out_cat(xs=generated_embeddings)

        return embedding, scores

class Encoder(nn.Module):
    def __init__(self, feat_dim, hid_dim, agg_gnn, gat_head, drop_out, num_levels):
        super(Encoder, self).__init__()
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.agg_gnn = agg_gnn
        self.gat_head = gat_head
        self.drop_out = drop_out
        self.num_levels = num_levels
        
        self.convs = nn.ModuleList()
        for level in range(self.num_levels):
            if level==0:
                if self.agg_gnn != 'GAT':
                    self.convs.append(GCNConv(self.feat_dim, self.hid_dim))
                else:
                    self.convs.append(GATConv(in_channels=self.feat_dim, 
                                                out_channels=(self.hid_dim//self.gat_head), 
                                                heads=self.gat_head,
                                                dropout=self.drop_out))
            else:
                if self.agg_gnn != 'GAT':
                    self.convs.append(GCNConv(self.hid_dim, self.hid_dim))
                else:
                    self.convs.append(GATConv(in_channels=self.hid_dim, 
                                                out_channels=(self.hid_dim//self.gat_head), 
                                                heads=self.gat_head,
                                                dropout=self.drop_out))
    
    def forward(self, x, level, edge_index, edge_weight):
        if self.agg_gnn != 'GAT':
            embedding = self.convs[level](x=x, edge_index=edge_index, edge_weight=edge_weight) # Z n*d
        else:
            embedding = self.convs[level](x=x, edge_index=edge_index) # Z n*d
        to_next = F.relu(embedding)
        
        return embedding, to_next


class Adaptive_pooling(nn.Module):
    def __init__(self, config, in_size, cluster_range, overlap, all_cluster, pooling_ratio):
        super(Adaptive_pooling, self).__init__()
        self.config = config
        self.in_size = in_size
        self.cluster_range = cluster_range
        self.overlap = overlap
        self.all_cluster = all_cluster
        self.pooling_ratio = pooling_ratio
        
        self.cluster_assignment = Cluster_assignment(self.cluster_range, 
                                                     device=self.config['device'])
        self.score_lin = nn.Linear(2*self.in_size, 1)
        self.gat_att = nn.Linear(2*self.in_size, 1)
        
    def forward(self, embedding, edge_index, edge_matrix, edge_matrix_weight, edge_weight=None, batch=None):
        # input: Z, A
        # output: _Z, _A, B, cluster_ids, batch
        N = embedding.size(0) # number of nodes
        
        # reconstruct graph where each edge's weight is the clossness between a pair of nodes
        # all values in [0, 1] and disgnal values are 1
        # A
        # normalize input embedding
        Z = F.normalize(embedding, p=2, dim=-1)
        connectivity_M = torch.mm(Z, Z.t())
        
        # structure loss
        x_pair = self.score_lin(torch.cat((embedding[edge_index[0]], embedding[edge_index[1]]), dim=-1))
        x_score = F.leaky_relu(x_pair)
        x_score = tg.utils.softmax(x_score, edge_index[0], num_nodes=embedding.shape[0])
        structure_M = tg.utils.to_dense_adj(edge_index=edge_index, edge_attr=x_score).squeeze(-1)[0]
        
        if self.config['fitness_mode']=='c':
            fitness = connectivity_M
        elif self.config['fitness_mode']=='s':
            fitness = structure_M
        elif self.config['fitness_mode']=='both_j':
            fitness = connectivity_M + structure_M
        elif self.config['fitness_mode']=='both_c':
            fitness = connectivity_M * structure_M
        
        # set up clusters with lamda hops
        # diagnal diagonal values are 0
        cluster_matrices = Generate_clusters(matrix=edge_matrix, 
                                             edge_score=fitness,
                                             cluster_range=self.cluster_range,
                                             threshold=self.config['edge_threshold']) # list of n*n
        
        # represent clusters within one matrix 
        cluster_matrix = torch.stack(cluster_matrices, dim=0).sum(0)
        cluster_matrix = (cluster_matrix > 0).float()
        
        # calculate the concentration of each cluster
        # make sure isolated nodes with 0 scores, therefore no activation function after it
        cluster_scores = self.cluster_assignment(fitness=fitness, 
                                                 cluster_matrices=cluster_matrices) # list of n*1
        
        # select clusters
        cluster_ids = Select_clusters(config=self.config, 
                                      fitness=fitness, 
                                      edge_matrix=edge_matrix,
                                      cluster_matrix=cluster_matrix,
                                      cluster_matrices=cluster_matrices,
                                      cluster_scores=cluster_scores,
                                      pooling_ratio=self.pooling_ratio,
                                      cluster_range=self.cluster_range,
                                      pick_all=self.all_cluster,
                                      overlap=self.overlap)
        
        if self.config['do_view']:
            # observe process
            a = fill_diagonal(A=fitness, value=0, device=fitness.device)
            f_min = a.min().data.cpu().numpy()[()]
            f_max = a.max().data.cpu().numpy()[()]
            c_min = cluster_scores.min().data.cpu().numpy()[()]
            c_max = cluster_scores.max().data.cpu().numpy()[()]    
            print(f'fitness min {f_min:.3f}, max {f_max:.3f}; '+
                  f'cluster min {c_min:.3f}, max {c_max:.3f}')
            f_mean = (a.sum()/torch.nonzero(a).size(0)).data.cpu().numpy()[()]
            c_mean = cluster_scores.mean().data.cpu().numpy()[()]
            b = cluster_scores[cluster_ids].clone()
            s_mean = b.mean().data.cpu().numpy()[()]
            print(f'fitness mean: {f_mean:.3f}, cluster mean: {c_mean:.3f}, selected cluster mean: {s_mean:.3f}')
        
        # maintaining graph connectivity
        # S (assignment_matrix), B
        S, _, S_w, S_w_orig, keeping_nodes, _S_ = Generate_assignment_matrix(cluster_matrix=cluster_matrix,
                                                                        cluster_ids=cluster_ids,
                                                                        fitness=fitness,
                                                                        do_view=self.config['do_view'])
        
#         B = S.clone()
        B = S_w.clone()
        
        # _X
        if self.config['pooling_mode']=='mean': 
            _embedding = torch.mm(S_w.t(), embedding)
        elif self.config['pooling_mode']=='att':
            # only available for non-overlapping case
            # attention-1
            # set up query and message
            query_M = fill_diagonal(A=S_w_orig, value=1, device=S_w_orig.device)
            if keeping_nodes.shape[0]>0:
                query_M[:, list(set(range(N))-set(torch.cat((cluster_ids, keeping_nodes), dim=-1).tolist()))]=0 # reduced nodes have no components
            else:
                query_M[:, list(set(range(N))-set(cluster_ids.tolist()))]=0
            pair = torch.nonzero(query_M>0).t()
            x_pair = self.gat_att(torch.cat((embedding[pair[0]], embedding[pair[1]]), dim=-1))
            x_score = F.leaky_relu(x_pair)
#             x_score = tg.utils.softmax(x_score, pair[0], num_nodes=embedding.shape[0])
            x_score = tg.utils.softmax(x_score, pair[1], num_nodes=embedding.shape[0])
            cluster_weight = tg.utils.to_dense_adj(edge_index=pair, edge_attr=x_score).squeeze(-1)[0]
            if cluster_weight.shape[0] < S_w.shape[0]:
                cluster_weight = F.pad(input=cluster_weight, 
                                       pad=(0, S_w.shape[0]-cluster_weight.shape[0], 0, S_w.shape[0]-cluster_weight.shape[0]), 
                                       mode='constant', value=0)
    
            _embedding = torch.mm(S_w.t(), cluster_weight)
            _embedding = torch.mm(_embedding, embedding)
        elif self.config['pooling_mode']=='max':
            _embedding = torch.mm(_S_.t(), embedding)

        # _A
#         edge_matrix = fill_diagonal(A=edge_matrix, value=1, device=edge_matrix.device)
        edge_matrix = fill_diagonal(A=edge_matrix_weight, value=1, device=edge_matrix.device)
        _edge_matrix_weight = torch.mm(torch.mm(S.t(), edge_matrix), S)
        
        _edge_matrix = _edge_matrix_weight.clone()
        _edge_matrix[_edge_matrix>0] = 1
        
        _edge_index, _edge_weight = tg.utils.dense_to_sparse(_edge_matrix_weight)
        
        return _embedding, _edge_matrix, _edge_matrix_weight, _edge_index, _edge_weight, B, cluster_ids, batch

class AHGNN_LP(nn.Module):
    def __init__(self, config, feat_dim):
        super(AHGNN_LP, self).__init__()
        self.config = config
        self.feat_dim = feat_dim
        self.agg_gnn = config['local_agg_gnn']
        self.hid_dim = config['hid_dim']
        self.num_levels = config['num_levels']
        self.output_mode = config['output_mode']
        
        self.encoder = Encoder(feat_dim=self.feat_dim,
                               hid_dim=self.hid_dim,
                               agg_gnn=self.agg_gnn,
                               gat_head=self.config['gat_head'], 
                               drop_out=self.config['drop_ratio'],
                               num_levels=self.num_levels)
        
        self.pools = nn.ModuleList()
        for idx in range(self.num_levels):
            self.pools.append(Adaptive_pooling(config=self.config,
                                               in_size=self.hid_dim,
                                               cluster_range=self.config['cluster_range'],
                                               overlap=self.config['overlap'],
                                               all_cluster=self.config['all_cluster'],
                                               pooling_ratio=self.config['pooling_ratio']))
        
        self.out_cat = Merge_xs(config=self.config,
                                mode=self.output_mode, 
                                dim=self.hid_dim,
                                num_levels=self.num_levels)
        
        self.last_gnn = GCNConv(self.hid_dim, self.hid_dim)
    
    def forward(self, data, epoch_id):
        x, edge_index, edge_M, batch = data.x, data.train_pos_edge_index, data.edge_matrix, data.batch
        
        if x.shape[0] > edge_index.max().item():
            edge_index = torch.cat((edge_index, torch.Tensor([[data.x.shape[0]-1], 
                                                              [data.x.shape[0]-1]]).long().to(edge_index.device)), dim=-1)
        edge_M = tg.utils.to_dense_adj(edge_index)[0]
        edge_M_weight = edge_M.clone()
        edge_weight = edge_M_weight[edge_M_weight>0]
        
        orig_edge_index = edge_index
        orig_edge_weight = edge_weight

        generated_embeddings = []
        recover_matrices = []
        As = []
        weighted_As = []

        for level in range(self.num_levels):
            # gnn embedding
            embedding, to_next = self.encoder(x, level, edge_index, edge_weight)
            if level==0:
                embedding_gnn = embedding.clone()
#                 p = self.orig_gnn(x=embedding, edge_index=edge_index, edge_weight=edge_weight)
            
            # _X, _A, _A_w, _edge_index, _edge_weight, B, cluster_ids, batch
            _input, _edge_M, _edge_M_weight, _edge_index, _edge_weight, recover_M, cluster_ids, batch = self.pools[level](embedding=to_next,
                                                                                                                          edge_index=edge_index,
                                                                                                                          edge_matrix=edge_M,
                                                                                                                          edge_matrix_weight=edge_M_weight,
                                                                                                                          edge_weight=edge_weight,
                                                                                                                          batch=batch)
            
            _input = F.normalize(_input, p=2, dim=1)

            if level>0:
                for idx, recover in enumerate(reversed(recover_matrices)):
                    embedding = torch.mm(recover, embedding)
                    
            generated_embeddings.append(embedding)
            recover_matrices.append(recover_M)
            As.append(edge_M)
            weighted_As.append(edge_M_weight)

            # assign the embedding as next level's input feature matrix
            x = _input
            edge_M = _edge_M
            edge_M_weight = _edge_M_weight
            edge_weight = _edge_weight
            edge_index = _edge_index
            
            # feature normalization
            x = x / x.sum(1, keepdim=True).clamp(min=1)
        
        embedding, scores = self.out_cat(data=data, xs=generated_embeddings)
        
        loss_kl = kl_loss(mu=embedding, logvar=embedding_gnn)
        
        embedding = self.last_gnn(x=embedding, 
                                  edge_index=orig_edge_index, 
                                  edge_weight=orig_edge_weight)
        
#         out = embedding
        out = F.normalize(embedding, p=2, dim=-1)
        
        return out, loss_kl, recover_matrices, scores

class AHGNN_NC(nn.Module):
    def __init__(self, config, feat_dim, out_dim):
        super(AHGNN_NC, self).__init__()
        self.config = config
        self.feat_dim = feat_dim
        self.agg_gnn = config['local_agg_gnn']
        self.hid_dim = config['hid_dim']
        self.out_dim = out_dim
        self.num_levels = config['num_levels']
        self.output_mode = config['output_mode']
        
        self.encoder = Encoder(feat_dim=self.feat_dim,
                               hid_dim=self.hid_dim,
                               agg_gnn=self.agg_gnn,
                               gat_head=self.config['gat_head'], 
                               drop_out=self.config['drop_ratio'],
                               num_levels=self.num_levels)
        
        self.pools = nn.ModuleList()
        for level in range(self.num_levels):
            self.pools.append(Adaptive_pooling(config=self.config,
                                               in_size=self.hid_dim,
                                               cluster_range=self.config['cluster_range'],
                                               overlap=self.config['overlap'],
                                               all_cluster=self.config['all_cluster'],
                                               pooling_ratio=self.config['pooling_ratio']))
        
        self.out_cat = Merge_xs(config=self.config,
                                mode=self.output_mode, 
                                dim=self.hid_dim,
                                num_levels=self.num_levels)
        
        self.last_gnn = GCNConv(self.hid_dim, self.out_dim)
    
    def forward(self, data, epoch_id):
        x, edge_index, edge_M, batch = data.x, data.edge_index, data.edge_matrix, data.batch
        edge_M_weight = edge_M.clone()
        edge_weight = edge_M_weight[edge_M_weight>0]
        
        orig_edge_index = edge_index
        orig_edge_weight = edge_weight

        generated_embeddings = []
        recover_matrices = []
        As = []
        weighted_As = []
        fitness = []

        for level in range(self.num_levels):
            # gnn embedding
            embedding, to_next = self.encoder(x, level, edge_index, edge_weight)

            if level==0:
                embedding_gnn = embedding.clone()
            
            # _X, _A, _A_w, _edge_index, _edge_weight, B, cluster_ids, batch
            _input, _edge_M, _edge_M_weight, _edge_index, _edge_weight, recover_M, cluster_ids, batch, fit = self.pools[level](embedding=to_next,
                                                                                                                          edge_index=edge_index,
                                                                                                                          edge_matrix=edge_M,
                                                                                                                          edge_matrix_weight=edge_M_weight,
                                                                                                                          edge_weight=edge_weight,
                                                                                                                          batch=batch)
            
            _input = F.normalize(_input, p=2, dim=1)

            if level>0:
                for idx, recover in enumerate(reversed(recover_matrices)):
                    embedding = torch.mm(recover, embedding)
            
            generated_embeddings.append(embedding)
            recover_matrices.append(recover_M)
            As.append(edge_M)
            weighted_As.append(edge_M_weight)
            fitness.append(fit)

            # assign the embedding as next level's input feature matrix
            x = _input
            edge_M = _edge_M
            edge_M_weight = _edge_M_weight
            edge_weight = _edge_weight
            edge_index = _edge_index
            
        embedding, scores = self.out_cat(data=data, xs=generated_embeddings)
        
        loss_kl = kl_loss(mu=embedding, logvar=embedding_gnn)
        loss_recon = recon_loss(z=embedding, pos_edge_index=orig_edge_index)
        
        embedding = self.last_gnn(x=embedding, 
                                  edge_index=orig_edge_index, 
                                  edge_weight=orig_edge_weight)
        
        out = F.log_softmax(embedding, dim=1)
        
        return out, loss_recon, loss_kl, recover_matrices, scores, fitness

class AHGNN_GC(nn.Module):
    def __init__(self, config, feat_dim, out_dim):
        super(AHGNN_GC, self).__init__()
        self.config = config
        self.feat_dim = feat_dim
        self.agg_gnn = config['local_agg_gnn']
        self.hid_dim = config['hid_dim']
        self.out_dim = out_dim
        self.num_levels = config['num_levels']
        self.output_mode = config['output_mode']
        
        self.encoder = Encoder(feat_dim=self.feat_dim,
                               hid_dim=self.hid_dim,
                               agg_gnn=self.agg_gnn,
                               gat_head=self.config['gat_head'], 
                               drop_out=self.config['drop_ratio'],
                               num_levels=self.num_levels)
        
        self.pools = nn.ModuleList()
        for idx in range(self.num_levels):
            self.pools.append(Adaptive_pooling(config=self.config,
                                               in_size=self.hid_dim,
                                               cluster_range=self.config['cluster_range'],
                                               overlap=self.config['overlap'],
                                               all_cluster=self.config['all_cluster'],
                                               pooling_ratio=self.config['pooling_ratio']))
        
        self.out_cat = Merge_xs(config=self.config,
                                mode=self.output_mode, 
                                dim=self.hid_dim,
                                num_levels=self.num_levels)
        
        self.last_gnn = GCNConv(self.hid_dim, self.hid_dim)
        
        self.lin1 = torch.nn.Linear(2*self.hid_dim, self.hid_dim)
        self.lin2 = torch.nn.Linear(self.hid_dim, self.hid_dim//2)
        self.lin3 = torch.nn.Linear(self.hid_dim//2, self.out_dim)
    
    def forward(self, data, epoch_id):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if x is None:
            x = torch.zeros(edge_index.max().item()+1, 128).to(edge_index.device)
        data.x = x
        
        if x.shape[0] > edge_index.max().item():
            edge_index = torch.cat((data.edge_index, torch.Tensor([[data.x.shape[0]-1], 
                                                                   [data.x.shape[0]-1]]).long().to(edge_index.device)), dim=-1)
        
        edge_M = tg.utils.to_dense_adj(edge_index)[0]
        edge_M_weight = edge_M.clone()
        edge_weight = edge_M_weight[edge_M_weight>0]
        
        orig_edge_index = edge_index
        orig_edge_weight = edge_weight

        generated_embeddings = []
        recover_matrices = []
        As = []
        weighted_As = []

        for level in range(self.num_levels):
            # gnn embedding
            embedding, to_next = self.encoder(x, level, edge_index, edge_weight)
            if level==0:
                embedding_gnn = embedding.clone()
#                 p = self.orig_gnn(x=embedding, edge_index=edge_index, edge_weight=edge_weight)
                x1 = torch.cat([gmp(embedding_gnn, batch), gap(embedding_gnn, batch)], dim=1)
            
            # _X, _A, _A_w, _edge_index, _edge_weight, B, cluster_ids, batch
            _input, _edge_M, _edge_M_weight, _edge_index, _edge_weight, recover_M, cluster_ids, batch = self.pools[level](embedding=to_next,
                                                                                                                          edge_index=edge_index,
                                                                                                                          edge_matrix=edge_M,
                                                                                                                          edge_matrix_weight=edge_M_weight,
                                                                                                                          edge_weight=edge_weight,
                                                                                                                          batch=batch)
            
            _input = F.normalize(_input, p=2, dim=1)

            if level>0:
                for idx, recover in enumerate(reversed(recover_matrices)):
                    embedding = torch.mm(recover, embedding)

            generated_embeddings.append(embedding)
            recover_matrices.append(recover_M)
            As.append(edge_M)
            weighted_As.append(edge_M_weight)

            # assign the embedding as next level's input feature matrix
            x = _input
            edge_M = _edge_M
            edge_M_weight = _edge_M_weight
            edge_weight = _edge_weight
            edge_index = _edge_index
        
#         embedding, scores = self.out_cat(data=data, xs=generated_embeddings)
#         x2 = torch.cat([gmp(embedding, batch), gap(embedding, batch)], dim=1)
        
#         loss_kl = kl_loss(mu=embedding, logvar=embedding_gnn)
#         loss_recon = recon_loss(z=embedding, pos_edge_index=orig_edge_index)
        
        embedding = self.last_gnn(x=embedding, 
                                  edge_index=orig_edge_index, 
                                  edge_weight=orig_edge_weight)
        x3 = torch.cat([gmp(embedding, batch), gap(embedding, batch)], dim=1)
        
#         x = F.relu(x1) + F.relu(x2) + F.relu(x3)
        x = F.relu(x1) + F.relu(x3)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.config['drop_ratio'], training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=self.config['drop_ratio'], training=self.training)
        
        out = F.log_softmax(x, dim=1)
        
#         return out, loss_recon, loss_kl, recover_matrices, scores
        return out, torch.Tensor([0]).to(device), torch.Tensor([0]).to(device), torch.Tensor([0]).to(device), torch.Tensor([0]).to(device)