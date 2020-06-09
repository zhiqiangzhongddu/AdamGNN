#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import random
import networkx as nx
import time
from copy import deepcopy
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, normalized_mutual_info_score
from load_dataset import *
from set_up_training import *
from prepare_data import *
from utils import *
import sys
from run import *
from model import Generate_high_order_adjacency_matrix, Encoder, Merge_xs

import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import torch_geometric as tg
from torch_geometric.nn import GraphConv, GCNConv, SAGEConv, GATConv, GINConv, JumpingKnowledge
from torch_geometric.nn.pool.topk_pool import topk

SEED = random.randint(1,10001)
# SEED = 4329
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline


# In[ ]:





# In[2]:


print(sys.argv)
data_name = sys.argv[1]
adam_lr = float(sys.argv[2])
num_train = int(sys.argv[3])
local_agg_gnn = sys.argv[4]
gat_head = int(sys.argv[5])
fitness_mode = sys.argv[6]
pooling_mode = sys.argv[7]
num_levels = int(sys.argv[8])
hid_dim = int(sys.argv[9])
overlap = eval(sys.argv[10])
cluster_range = int(sys.argv[11])
drop_ratio = float(sys.argv[12])
loss_mode = sys.argv[13]

# data_name = 'disease_1' # emails cora citeseer pubmed cs phisic computers photo 
# adam_lr = 0.01
# num_train = 70
# local_agg_gnn = 'GCN' # GCN SAGE GIN
# gat_head = 0
# fitness_mode = 'both_c' # c: connect s: structure both_j both_c
# pooling_mode = 'att' # mean att max
# num_levels = 3 # 1, 2, 3
# hid_dim = 64
# overlap = True
# cluster_range = 1
# drop_ratio = 0.9
# loss_mode = 'all' # nc recon kl all

output_mode = 'ATT' # ATT GCN GAT MEAN MAX LINEAR LSTM NONE
edge_threshold = 0
all_cluster = True
pooling_ratio = 0.1


# In[ ]:





# In[3]:


batch_size = None
dropout = True # no necessary
relu = True # optional for shallow layers
early_stop = 100
use_attribute = True
do_view = False

#################Parameters for model#################
loc_time = time.localtime()
model_config={'data_name': data_name,
              'num_train': num_train,
              'num_epoch': 1001, # 2001
              'batch_size': batch_size,
              'adam_lr': adam_lr, # 1e-2, 5e-3
              'l2_regularization': 5e-4, #5e-4, 7e-4
              'dropout': dropout, 
              'drop_ratio': drop_ratio,
              'relu': relu,
              'local_agg_gnn': local_agg_gnn,
              'gat_head': gat_head,
              'fitness_mode': fitness_mode,
              'pooling_mode': pooling_mode,
              'output_mode': output_mode,
              'num_levels': num_levels,
              'hid_dim': hid_dim,
              'overlap': overlap,
              'edge_threshold': edge_threshold,
              'all_cluster': all_cluster,
              'pooling_ratio': pooling_ratio,
              'cluster_range': cluster_range,
              'loss_mode': loss_mode,
              'verbose': 1, 
              'early_stop': early_stop,
              'use_attribute': use_attribute,
              'do_view': do_view,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model',
              'alias':'test_model_{}_{}_{}_{}_{}'.format(loc_time[0], loc_time[1], loc_time[2], loc_time[3], loc_time[4])}

if torch.cuda.is_available():
    model_config['use_cuda'] = True
else:
    model_config['use_cuda'] = False

print('Parameters of Model are:')
for _ in model_config:
    print(_, model_config[_])
    
# set up device
device = torch.device('cuda:'+str(0) if model_config['use_cuda'] else 'cpu')
model_config['device'] = device


# In[ ]:





# In[ ]:





# In[4]:


# load dataset
ls_data = get_NC_dataset(dataset_name=data_name,
                         use_features=use_attribute,
                         SEED=SEED)
# set up training mode
ls_data = set_up_NC_training(ls_data=ls_data, 
                             num_train=num_train, 
                             SEED=SEED)
# prepare data
ls_data = prepare_NC_data(ls_data=ls_data, 
                          batch_size=batch_size, 
                          device=device, 
                          SEED=SEED)

norm = tg.transforms.NormalizeFeatures()
for idx, data in enumerate(ls_data):
    ls_data[idx] = norm(data)


# In[ ]:





# In[ ]:





# In[ ]:





# ## set up model

# In[5]:


def Generate_clusters(matrix, edge_score, cluster_range, threshold):
    matrix = fill_diagonal(A=matrix, value=0, device=matrix.device)
    matrix[torch.where(edge_score < threshold)] = 0
    
    cluster_matrices = []
    for order in range(1, cluster_range+1):
        cluster_matrices.append(Generate_high_order_adjacency_matrix(A=matrix, order=order))
    
    return cluster_matrices


# In[ ]:





# In[6]:


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


# In[ ]:





# In[7]:


def Select_clusters(config, fitness, edge_matrix, cluster_matrix, cluster_matrices, cluster_scores, pooling_ratio, cluster_range, pick_all, overlap):
    # input: cluster_matrix, cluster_scores, pooling_ratio
    # output: cluster_fitness
    num_nodes = cluster_matrix.shape[0]
    if overlap:
        _cluster_matrix = cluster_matrices[0]
        local_extrema = (cluster_scores.expand(num_nodes, num_nodes)).t()                            - torch.mul(_cluster_matrix, cluster_scores.expand(num_nodes, num_nodes)) > 0
    else:
        cluster_matrices = Generate_clusters(matrix=edge_matrix,
                                             edge_score=fitness, 
                                             cluster_range=cluster_range*2,
                                             threshold=config['edge_threshold'])
        # represent clusters within one matrix 
        _cluster_matrix = torch.stack(cluster_matrices).sum(0)
        _cluster_matrix = (_cluster_matrix > 0.5).float().to(_cluster_matrix.device)
        local_extrema = (cluster_scores.expand(num_nodes, num_nodes)).t()                            - torch.mul(_cluster_matrix, cluster_scores.expand(num_nodes, num_nodes)) > 0
    
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


# In[ ]:





# In[8]:


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
    keeping_nodes = torch.tensor(list(set(range(num_nodes))-                                      set(cluster_ids.tolist())-                                      set(reduced_nodes.tolist()))).to(cluster_ids.device)
    
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


# In[ ]:





# In[9]:


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


# In[ ]:





# In[10]:


class AHGNN(nn.Module):
    def __init__(self, config, feat_dim, out_dim):
        super(AHGNN, self).__init__()
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

        for level in range(self.num_levels):
            # gnn embedding
            embedding, to_next = self.encoder(x, level, edge_index, edge_weight)

            if level==0:
                embedding_gnn = embedding.clone()
            
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
            
        embedding, scores = self.out_cat(data=data, xs=generated_embeddings)
        
        loss_kl = kl_loss(mu=embedding, logvar=embedding_gnn)
        loss_recon = recon_loss(z=embedding, pos_edge_index=orig_edge_index)
        
        embedding = self.last_gnn(x=embedding, 
                                  edge_index=orig_edge_index, 
                                  edge_weight=orig_edge_weight)
        
        out = F.log_softmax(embedding, dim=1)
        
        return out, loss_recon, loss_kl, recover_matrices, scores


# In[ ]:





# ## experiments

# In[11]:


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
#         nn.init.uniform_(m.weight.data)
#         torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
#         torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        pass


# In[ ]:





# In[12]:


model = AHGNN(config=model_config,
              feat_dim=ls_data[0].x.shape[1],
              out_dim=ls_data[0].y.unique().size(0))

# release gpu memory
torch.cuda.empty_cache()

model = model.to(device)
model.apply(weights_init)
print(model)

optimizer = torch.optim.Adam(model.parameters(), 
                             lr=model_config['adam_lr'],
                             weight_decay=model_config['l2_regularization'])


# In[ ]:





# In[13]:


NC_run(model=model, 
       optimizer=optimizer, 
       ls_data=ls_data, 
       model_config=model_config, 
       device=device, 
       SEED=SEED)


# In[ ]:





# In[ ]:




