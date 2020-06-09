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
from model import Generate_high_order_adjacency_matrix, Encoder, Merge_xs, Adaptive_pooling

import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import torch_geometric as tg
from torch_geometric.nn import GraphConv, GCNConv, SAGEConv, GATConv, GINConv, JumpingKnowledge, GAE, InnerProductDecoder
from torch_geometric.nn.pool.topk_pool import topk

SEED = random.randint(1,10001)
# SEED = 4329 # emails
# SEED = 2523
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline


# In[ ]:





# In[5]:


print(sys.argv)
data_name = sys.argv[1]
adam_lr = float(sys.argv[2])
ratio_train = int(sys.argv[3])
local_agg_gnn = sys.argv[4]
gat_head = int(sys.argv[5])
fitness_mode = sys.argv[6] # c: connect s: structure both_j both_c
pooling_mode = sys.argv[7]
num_levels = int(sys.argv[8])
hid_dim = int(sys.argv[9])
overlap = eval(sys.argv[10])
cluster_range = int(sys.argv[11])
drop_ratio = float(sys.argv[12])
loss_mode = sys.argv[13]

# data_name = 'dphi_1' # emails cora citeseer pubmed cs phisic computers photo 
# adam_lr = 0.01
# ratio_train = 90
# num_class = 5
# local_agg_gnn = 'GCN' # GCN SAGE GIN
# gat_head = 0
# fitness_mode = 'both_c' # c: connect s: structure both_j both_c
# pooling_mode = 'att' # mean att max
# num_levels = 4 # 1, 2, 3
# hid_dim = 64
# overlap = True
# cluster_range = 1
# drop_ratio = 0.9
# loss_mode = 'kl' # lp kl

output_mode = 'ATT' # ATT GCN GAT MEAN MAX LINEAR LSTM NONE
edge_threshold = 0
all_cluster = True
pooling_ratio = 0.1


# In[ ]:





# In[6]:


batch_size = None
dropout = True # no necessary
relu = True # optional for shallow layers
early_stop = 100
use_attribute = True
do_view = False

#################Parameters for model#################
loc_time = time.localtime()
model_config={'data_name': data_name,
              'ratio_train': ratio_train, 
              'num_epoch': 1001, # 2001
              'batch_size': batch_size,
              'optimizer': 'adam', # adam sgd
              'adam_lr': adam_lr, # 1e-4
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





# In[7]:


# load dataset
ls_data = get_LP_dataset(dataset_name=data_name,
                         use_features=use_attribute,
                         SEED=SEED)

# # set up training mode
# ls_data = set_up_LP_training_pyg(ls_data=ls_data,
#                              ratio_val=0.1,
#                              ratio_test=0.1,
#                              SEED=SEED)
# # prepare data
# ls_data = prepare_LP_data_pyg(ls_data=ls_data,
#                           batch_size=batch_size,
#                           device=device)

# set up training mode
graphs, ls_df_train, ls_df_valid, ls_df_test = set_up_LP_training(ls_data=ls_data,
                                                                  num_train=ratio_train, 
                                                                  SEED=SEED)
# prepare data
ls_data = prepare_LP_data(ls_data=ls_data,
                          ls_df_train=ls_df_train, 
                          ls_df_valid=ls_df_valid, 
                          ls_df_test=ls_df_test, 
                          batch_size=batch_size,
                          device=device,
                          SEED=SEED)

print(f'train {ls_data[0].train_pos_edge_index.shape[1]}, valid {ls_data[0].val_pos_edge_index.shape[1]}, test {ls_data[0].test_pos_edge_index.shape[1]}')


# In[ ]:





# ## set up model

# In[8]:


class AHGNN(nn.Module):
    def __init__(self, config, feat_dim):
        super(AHGNN, self).__init__()
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


# In[ ]:





# ## experiments

# In[9]:


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
#         m.weight.data = torch.nn.init.uniform_(m.weight.data)
#         m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
#         m.weight.data = torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        pass


# In[ ]:





# In[10]:


encoder = AHGNN(config=model_config,
              feat_dim=ls_data[0].x.shape[1])
model = GAE(encoder=encoder, decoder=InnerProductDecoder())
model = model.to(device)
model.apply(weights_init)
print(model)


# In[ ]:





# In[11]:


optimizer = torch.optim.Adam(model.parameters(), 
                             lr=model_config['adam_lr'],
                             weight_decay=model_config['l2_regularization'])

LP_run(model=model, 
       optimizer=optimizer, 
       ls_data=ls_data, 
       model_config=model_config, 
       n_clusters=ls_data[0].y.unique().shape[0],
       device=device, 
       SEED=SEED)


# In[ ]:





# In[ ]:





# In[ ]:




