#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
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

from tensorboardX import SummaryWriter
import torch
import torch_geometric as tg
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, GraphUNet, InnerProductDecoder, GAE
from torch_geometric.utils import dropout_adj

SEED = random.randint(1,10001)
# SEED = 4329 # emails
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline


# In[2]:


print(sys.argv)
data_name = sys.argv[1]
ratio_train = int(sys.argv[2])
num_class = int(sys.argv[3])
baseline_gnn = sys.argv[4]
num_gnn_layers = int(sys.argv[5])
hid_dim = int(sys.argv[6])
drop_ratio = float(sys.argv[7])

# data_name = 'dphi_1'
# ratio_train = 80
# num_class = 5
# baseline_gnn = 'GCN' # GCN SAGE GAT GIN GUNET
# num_gnn_layers = 2 # 1, 2, 3
# hid_dim = 64
# drop_ratio = 0.5


# In[3]:


feature_pre = False
dropout = True # no necessary
relu = True # optional for shallow layers
batch_size = None
early_stop = 50
use_attribute = True

#################Parameters for model#################
loc_time = time.localtime()
model_config={'data_name': data_name,
              'ratio_train': ratio_train, 
              'num_epoch': 501, # 2001
              'batch_size': batch_size,
              'num_gnn_layers': num_gnn_layers,
              'optimizer': 'adam', # adam sgd
              'adam_lr': 1e-2, # 1e-4
              'l2_regularization': 5e-4, # 0 0.0000001, 5e-4
              'dropout': dropout, 
              'drop_ratio': drop_ratio,
              'relu': relu,
              'feature_pre': feature_pre,
              'baseline_gnn': baseline_gnn,
              'verbose': 1, 
              'use_attribute': use_attribute,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model',
              'alias':'test_model_{}_{}_{}_{}_{}'.format(loc_time[0], loc_time[1], loc_time[2], loc_time[3], loc_time[4])}

model_config['gnn_layers']=[hid_dim]*(num_gnn_layers+1)
    
if torch.cuda.is_available():
    model_config['use_cuda'] = True
else:
    model_config['use_cuda'] = False

print('Parameters of Model are:')
for _ in model_config:
    print(_, model_config[_])
    
# set up device
device = torch.device('cuda:'+str(0) if model_config['use_cuda'] else 'cpu')


# In[ ]:





# In[4]:


# load dataset
ls_data = get_LP_dataset(dataset_name=data_name,
                         use_features=use_attribute,
                         SEED=SEED)

# set up training mode
ls_data = set_up_LP_training_pyg(ls_data=ls_data,
                             ratio_val=(100-ratio_train)/100/2,
                             ratio_test=(100-ratio_train)/100/2,
                             SEED=SEED)
# prepare data
ls_data = prepare_LP_data_pyg(ls_data=ls_data,
                          batch_size=batch_size,
                          device=device)

# # set up training mode
# graphs, ls_df_train, ls_df_valid, ls_df_test = set_up_LP_training(ls_data=ls_data,
#                                                                   num_train=ratio_train, 
#                                                                   SEED=SEED)
# # prepare data
# ls_data = prepare_LP_data(ls_data=ls_data,
#                           ls_df_train=ls_df_train, 
#                           ls_df_valid=ls_df_valid, 
#                           ls_df_test=ls_df_test, 
#                           batch_size=batch_size,
#                           device=device,
#                           SEED=SEED)

print(f'train {ls_data[0].train_pos_edge_index.shape[1]}, valid {ls_data[0].val_pos_edge_index.shape[1]}, test {ls_data[0].test_pos_edge_index.shape[1]}')


# In[ ]:





# ## set up model

# In[5]:


class Baseline_GNN (torch.nn.Module):
    def __init__(self, config, feat_dim, node_num):
        super(Baseline_GNN, self).__init__()
        self.config = config
        self.feat_dim = feat_dim
        self.node_num = node_num
        
        self.gnn_layers = torch.nn.ModuleList()
        self.config['gnn_layers'][0] = self.feat_dim
        
        if self.config['baseline_gnn'] == 'GCN':
            for idx, (in_size, out_size) in enumerate(zip(self.config['gnn_layers'][:-1], self.config['gnn_layers'][1:])):
                self.gnn_layers.append(GCNConv(in_size, out_size))
        elif self.config['baseline_gnn'] == 'SAGE':
            for idx, (in_size, out_size) in enumerate(zip(self.config['gnn_layers'][:-1], self.config['gnn_layers'][1:])):
                self.gnn_layers.append(SAGEConv(in_size, out_size))
        elif self.config['baseline_gnn'] == 'GAT':
            for idx, (in_size, out_size) in enumerate(zip(self.config['gnn_layers'][:-1], self.config['gnn_layers'][1:])):
                self.gnn_layers.append(GATConv(in_size, out_size, heads=1))
        elif self.config['baseline_gnn'] == 'GIN':
            self.lnn_layers = torch.nn.ModuleList()
            for idx, (in_size, out_size) in enumerate(zip(self.config['gnn_layers'][:-1], self.config['gnn_layers'][1:])):
                self.lnn_layers.append(torch.nn.Linear(in_size, out_size))
                self.gnn_layers.append(GINConv(self.lnn_layers[idx]))
        elif self.config['baseline_gnn'] == 'GUNET':
            if node_num < 2000:
                pool_ratios = [200 / node_num, 0.5]
            else:
                pool_ratios = [2000 / node_num, 0.5]
            self.unet = GraphUNet(feat_dim, 32, self.config['gnn_layers'][-1], 
                                  depth=self.config['num_gnn_layers'], 
                                  pool_ratios=pool_ratios)
                
    def forward(self, x, edge_index):
        if self.config['baseline_gnn']=='GUNET':
            edge_index, _ = dropout_adj(edge_index, p=self.config['drop_ratio'], 
                                        force_undirected=True,
                                        num_nodes=x.shape[0], training=self.training)
            embed = torch.nn.functional.dropout(x, p=self.config['drop_ratio'], training=self.training)
            embed = self.unet(embed, edge_index)
            
        else:
            embed = x
            for idx, _ in enumerate(range(len(self.gnn_layers))):
                if idx != len(self.gnn_layers)-1:
                    # same level
                    embed = F.relu(self.gnn_layers[idx](embed, edge_index)) # Note: optional!
                    embed = F.dropout(embed, p=self.config['drop_ratio'], training=self.training)
                else:
                    embed = self.gnn_layers[idx](embed, edge_index)

#         embed = torch.nn.functional.normalize(embed, p=2, dim=-1)
        return embed


# In[ ]:





# ## experiments

# In[6]:


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
#         m.weight.data = torch.nn.init.uniform_(m.weight.data)
#         m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
#         m.weight.data = torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        pass


# In[ ]:





# In[7]:


encoder = Baseline_GNN(config=model_config, feat_dim=ls_data[0].x.shape[1], node_num=ls_data[0].x.shape[0])
model = tg.nn.GAE(encoder=encoder, decoder=InnerProductDecoder()).to(device)
model.apply(weights_init)
print(model)


# In[ ]:





# In[8]:


optimizer = torch.optim.Adam(model.parameters(), 
                             lr=model_config['adam_lr'],)
#                              weight_decay=model_config['l2_regularization'])


# In[ ]:





# In[9]:


writer = SummaryWriter(logdir='runs/{}'.format(model_config['alias']))
writer.add_text('config', str(model_config), 0)

test_results_auc = []
valid_results_auc = []
test_results_ap = []
valid_results_ap = []

for epoch_id, epoch in enumerate(range(model_config['num_epoch'])):
    start_epoch = time.time()
    if epoch_id % model_config['verbose'] == 0:
        print('\nEpoch {} starts !'.format(epoch_id))
        print('-' * 80)
    total_loss = 0
    
    for idx, data in enumerate(ls_data):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.train_pos_edge_index)
        loss = model.recon_loss(z, data.train_pos_edge_index)

        # update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.cpu().item()

    # write epoch info
    writer.add_scalar('model/loss', total_loss, epoch_id)
    
    # evaluate epoch
    model.eval()
    for idx, data in enumerate(ls_data):
        z = model.encode(data.x, data.train_pos_edge_index)
        
#         train_auc, train_ap = model.test(z, data.train_pos_edge_index, data.train_neg_edge_index)
        val_auc, val_ap = model.test(z, data.val_pos_edge_index, data.val_neg_edge_index)
        test_auc, test_ap = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
        
    if epoch_id % model_config['verbose'] == 0:
        print('Evluating Epoch {}, time {:.3f}'.format(epoch_id, time.time()-start_epoch))
        print('valid ROC-AUC = {:.4f}, Test ROC-AUC = {:.4f}'.format(val_auc, test_auc))
        print('valid AP = {:.4f}, Test AP = {:.4f}'.format(val_ap, test_ap))
        
        valid_results_auc.append(val_auc)
        valid_results_ap.append(val_ap)
        test_results_auc.append(test_auc)
        test_results_ap.append(test_ap)
        
        print('best valid AUC performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.              format(max(valid_results_auc),
                     test_results_auc[valid_results_auc.index(max(valid_results_auc))],\
                      model_config['verbose']*valid_results_auc.index(max(valid_results_auc))))
        print('best valid AP performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.              format(max(valid_results_ap),
                     test_results_ap[valid_results_ap.index(max(valid_results_ap))],\
                      model_config['verbose']*valid_results_ap.index(max(valid_results_ap))))
        
        idx_1 = valid_results_auc.index(max(valid_results_auc))
        idx_2 = valid_results_ap.index(max(valid_results_ap))
        if (idx_1*model_config['verbose']+early_stop < epoch_id) and (idx_2*model_config['verbose']+early_stop < epoch_id):
            break


# In[ ]:





# In[ ]:


import datetime
now = datetime.datetime.now()

path = f"../output/LP_{data_name}_baseline_train_{ratio_train}_n_class_{num_class}"+        f"_{baseline_gnn}_layer_{num_gnn_layers}_hid_{hid_dim}"+        f"_drop_{drop_ratio}.dat"
print(path)
with open(path,"a+") as f:
    f.write('\n'+str(now.year)+'-'+str(now.month)+'-'+str(now.day)+'-'+
            str(now.hour)+'-'+str(now.minute)+'-'+str(now.second)+'-'+str(now.microsecond)+' '+
            str(test_results_auc[valid_results_auc.index(max(valid_results_auc))])+' '+
            str(test_results_ap[valid_results_ap.index(max(valid_results_ap))])+' '+
            str(SEED))


# In[ ]:





# In[ ]:





# In[ ]:




