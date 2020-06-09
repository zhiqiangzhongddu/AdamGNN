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
from torch import nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, GraphUNet
from torch_geometric.utils import dropout_adj

SEED = random.randint(1,10001)
# SEED = 90
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline


# In[2]:


print(sys.argv)
data_name = sys.argv[1]
num_train = int(sys.argv[2])
baseline_gnn = sys.argv[3]
num_gnn_layers = int(sys.argv[4])
hid_dim = int(sys.argv[5])
drop_ratio = float(sys.argv[6])

# data_name = 'emails'
# num_train = 80
# baseline_gnn = 'GUNET' # GCN SAGE GAT GIN GUNET
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
              'num_epoch': 501, # 2001
              'batch_size': batch_size,
              'optimizer': 'adam', # adam sgd
              'adam_lr': 1e-2, # 1e-4
              'l2_regularization': 5e-4, # 0 0.0000001, 5e-4
              'dropout': dropout, 
              'drop_ratio': drop_ratio,
              'relu': relu,
              'feature_pre': feature_pre,
              'baseline_gnn': baseline_gnn,
              'verbose': 10,
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
device = torch.device('cuda' if model_config['use_cuda'] else 'cpu')


# In[4]:


# load dataset
ls_data = get_NC_dataset(dataset_name=data_name,
                         use_features=use_attribute,
                         SEED=SEED)
ls_data = set_up_NC_training(ls_data=ls_data, 
                             num_train=num_train, 
                             SEED=SEED)
ls_data = prepare_NC_data(ls_data=ls_data, 
                          batch_size=batch_size, 
                          device=device, 
                          SEED=SEED)


# In[ ]:





# In[ ]:





# ## set up model

# In[5]:


class Baseline_GNN (nn.Module):
    def __init__(self, config, feat_dim, node_num, out_dim):
        super(Baseline_GNN, self).__init__()
        self.config = config
        self.feat_dim = feat_dim
        self.node_num = node_num
        self.out_dim = out_dim
        
        self.config['gnn_layers'][0] = self.feat_dim
        self.config['gnn_layers'][-1] = self.out_dim
        
        self.gnn_layers = torch.nn.ModuleList()
        if self.config['baseline_gnn'] == 'GCN':
            for idx, (in_size, out_size) in enumerate(zip(self.config['gnn_layers'][:-1], self.config['gnn_layers'][1:])):
                self.gnn_layers.append(SAGEConv(in_size, out_size))
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
            if ls_data[0].num_nodes < 2000:
                pool_ratios = [400 / ls_data[0].num_nodes, 0.5]
            else:
                pool_ratios = [2000 / ls_data[0].num_nodes, 0.5]
            self.unet = GraphUNet(ls_data[0].x.shape[1], hid_dim, self.config['gnn_layers'][-1], depth=num_gnn_layers, pool_ratios=pool_ratios)
            
    def forward(self, data):
        if self.config['baseline_gnn']=='GUNET':
            edge_index, _ = dropout_adj(data.edge_index, p=self.config['drop_ratio'], 
                                        force_undirected=True,
                                        num_nodes=data.x.shape[0], training=self.training)
            embed = F.dropout(data.x, p=self.config['drop_ratio'], training=self.training)
            embed = self.unet(embed, edge_index)
        else:
            embed, edge_index = data.x, data.edge_index

            for idx, _ in enumerate(range(len(self.gnn_layers))):
                if idx != len(self.gnn_layers)-1:
                    # same level
                    embed = self.gnn_layers[idx](embed, edge_index)
                    if self.config['relu']:
                        embed = F.relu(embed) # Note: optional!
                    if self.config['dropout']:
                        embed = F.dropout(embed, p=self.config['drop_ratio'], training=self.training)
                else:
                    embed = self.gnn_layers[idx](embed, edge_index)
    
        out = F.log_softmax(embed, dim=1)
        return out


# In[ ]:





# ## experiments

# In[6]:


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
#         nn.init.uniform_(m.weight.data)
#         torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
#         torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        pass


# In[7]:


model = Baseline_GNN(config=model_config, 
                     feat_dim=ls_data[0].x.shape[1], 
                     node_num=ls_data[0].x.shape[0],
                     out_dim=ls_data[0].y.unique().size(0))
model = model.to(device)
model.apply(weights_init)
print(model)


# In[ ]:





# In[8]:


optimizer = torch.optim.Adam(model.parameters(), 
                             lr=model_config['adam_lr'],
                             weight_decay=model_config['l2_regularization'])


# In[9]:


writer = SummaryWriter(logdir='runs/{}'.format(model_config['alias']))
writer.add_text('config', str(model_config), 0)

loss_func = torch.nn.NLLLoss()

test_results_auc = []
valid_results_auc = []
test_results_accuracy = []
valid_results_accuracy = []
test_results_f1_micro = []
valid_results_f1_micro = []
test_results_f1_macro = []
valid_results_f1_macro = []
test_results_nmi = []
valid_results_nmi = []

for epoch_id, epoch in enumerate(range(model_config['num_epoch'])):
    start_epoch = time.time()
    if epoch_id % model_config['verbose'] == 0:
        print('Epoch {} starts !'.format(epoch_id))
        print('-' * 80)
    total_loss = 0
    
    for idx, data in enumerate(ls_data):
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        labels_tensor = data.y
        
        model.train()
        optimizer.zero_grad()
        out = model.forward(data)
        preds_train = out[train_mask]
        loss = loss_func(preds_train, labels_tensor[train_mask])

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
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        labels = data.y.data.cpu().numpy()
        
        out = model.forward(data)
       
        preds_train = out[train_mask]
        preds_valid = out[val_mask]
        preds_test = out[test_mask]
        
    if epoch_id % model_config['verbose'] == 0:
        epoch_train_results_accuracy = accuracy_score(y_pred=np.argmax(preds_train.data.cpu().numpy(),  axis=1), 
                                                      y_true=labels[train_mask.data.cpu().numpy()])
        epoch_valid_results_accuracy = accuracy_score(y_pred=np.argmax(preds_valid.data.cpu().numpy(),  axis=1), 
                                                      y_true=labels[val_mask.data.cpu().numpy()])
        epoch_test_results_accuracy = accuracy_score(y_pred=np.argmax(preds_test.data.cpu().numpy(),  axis=1), 
                                                     y_true=labels[test_mask.data.cpu().numpy()])
        
        epoch_train_results_f1_micro = f1_score(y_pred=np.argmax(preds_train.data.cpu().numpy(), axis=1), 
                                                y_true=labels[train_mask.data.cpu().numpy()], average='micro')
        epoch_valid_results_f1_micro = f1_score(y_pred=np.argmax(preds_valid.data.cpu().numpy(), axis=1), 
                                                y_true=labels[val_mask.data.cpu().numpy()], average='micro')
        epoch_test_results_f1_micro = f1_score(y_pred=np.argmax(preds_test.data.cpu().numpy(), axis=1), 
                                               y_true=labels[test_mask.data.cpu().numpy()], average='micro')
        
        epoch_train_results_f1_macro = f1_score(y_pred=np.argmax(preds_train.data.cpu().numpy(), axis=1), 
                                                y_true=labels[train_mask.data.cpu().numpy()], average='macro')
        epoch_valid_results_f1_macro = f1_score(y_pred=np.argmax(preds_valid.data.cpu().numpy(), axis=1), 
                                                y_true=labels[val_mask.data.cpu().numpy()], average='macro')
        epoch_test_results_f1_macro = f1_score(y_pred=np.argmax(preds_test.data.cpu().numpy(), axis=1), 
                                                 y_true=labels[test_mask.data.cpu().numpy()], average='macro')
        
        epoch_train_results_nmi = normalized_mutual_info_score(labels_pred=np.argmax(preds_train.data.cpu().numpy(), axis=1), 
                                                               labels_true=labels[train_mask.data.cpu().numpy()])
        epoch_valid_results_nmi = normalized_mutual_info_score(labels_pred=np.argmax(preds_valid.data.cpu().numpy(), axis=1), 
                                                               labels_true=labels[val_mask.data.cpu().numpy()])
        epoch_test_results_nmi = normalized_mutual_info_score(labels_pred=np.argmax(preds_test.data.cpu().numpy(), axis=1), 
                                                              labels_true=labels[test_mask.data.cpu().numpy()])
        
        print('Evluating Epoch {}, time {:.3f}'.format(epoch_id, time.time()-start_epoch))
        print('train Accuracy = {:.4f}, valid Accuracy = {:.4f}, Test Accuracy = {:.4f}'.format(epoch_train_results_accuracy,
                                                                                                 epoch_valid_results_accuracy,
                                                                                                 epoch_test_results_accuracy))
        print('train Micro-F1 = {:.4f}, valid Micro-F1 = {:.4f}, Test Micro-F1 = {:.4f}'.format(epoch_train_results_f1_micro,
                                                                                                epoch_valid_results_f1_micro,
                                                                                                epoch_test_results_f1_micro))
        print('train Macro-F1 = {:.4f}, valid Macro-F1 = {:.4f}, Test Macro-F1 = {:.4f}'.format(epoch_train_results_f1_macro,
                                                                                                epoch_valid_results_f1_macro,
                                                                                                epoch_test_results_f1_macro))
        print('train NMI = {:.4f}, valid NMI = {:.4f}, Test NMI = {:.4f}'.format(epoch_train_results_nmi,
                                                                                 epoch_valid_results_nmi,
                                                                                 epoch_test_results_nmi))
        
        valid_results_accuracy.append(epoch_valid_results_accuracy)
        test_results_accuracy.append(epoch_test_results_accuracy)
        valid_results_f1_micro.append(epoch_valid_results_f1_micro)
        test_results_f1_micro.append(epoch_test_results_f1_micro)
        valid_results_f1_macro.append(epoch_valid_results_f1_macro)
        test_results_f1_macro.append(epoch_test_results_f1_macro)
        valid_results_nmi.append(epoch_valid_results_nmi)
        test_results_nmi.append(epoch_test_results_nmi)
        print('best valid Accuracy performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.              format(max(valid_results_accuracy),
                     test_results_accuracy[valid_results_accuracy.index(max(valid_results_accuracy))],\
                      model_config['verbose']*valid_results_accuracy.index(max(valid_results_accuracy))))
        print('best valid Micro-F1 performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.              format(max(valid_results_f1_micro),
                     test_results_f1_micro[valid_results_f1_micro.index(max(valid_results_f1_micro))],\
                      model_config['verbose']*valid_results_f1_micro.index(max(valid_results_f1_micro))))
        print('best valid Macro-F1 performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.              format(max(valid_results_f1_macro),                     test_results_f1_macro[valid_results_f1_macro.index(max(valid_results_f1_macro))],                     model_config['verbose']*valid_results_f1_macro.index(max(valid_results_f1_macro))))
        print('best valid NMI performance is {:.4f}, best test performance is {:.4f} and epoch_id is {}'.              format(max(valid_results_nmi),                     test_results_nmi[valid_results_nmi.index(max(valid_results_nmi))],                     model_config['verbose']*valid_results_nmi.index(max(valid_results_nmi))))
        
        idx_1 = valid_results_accuracy.index(max(valid_results_accuracy))
        idx_2 = valid_results_f1_micro.index(max(valid_results_f1_micro))
        idx_3 = valid_results_f1_macro.index(max(valid_results_f1_macro))
        idx_4 = valid_results_nmi.index(max(valid_results_nmi))
        if (idx_1*model_config['verbose']+early_stop < epoch_id)&(idx_2*model_config['verbose']+early_stop < epoch_id)&(idx_3*model_config['verbose']+early_stop < epoch_id)&(idx_4*model_config['verbose']+early_stop < epoch_id):
            break


# In[ ]:





# In[ ]:





# In[ ]:


import datetime
now = datetime.datetime.now()

path = f"../output/NC_data_{data_name}_baseline_num_train_{num_train}"+        f"_gnn_{baseline_gnn}_layers_{num_gnn_layers}_hid_dim_{hid_dim}"+        f"_drop_ratio_{drop_ratio}_results.dat"
print(path)
with open(path,"a+") as f:
    f.write('\n'+str(now.year)+'-'+str(now.month)+'-'+str(now.day)+'-'+
            str(now.hour)+'-'+str(now.minute)+'-'+str(now.second)+'-'+str(now.microsecond)+' '+
            str(test_results_accuracy[valid_results_accuracy.index(max(valid_results_accuracy))])+' '+
            str(test_results_f1_micro[valid_results_f1_micro.index(max(valid_results_f1_micro))])+' '+
            str(test_results_f1_macro[valid_results_f1_macro.index(max(valid_results_f1_macro))])+' '+
            str(test_results_nmi[valid_results_nmi.index(max(valid_results_nmi))])+' '+
            str(SEED))


# In[ ]:





# In[ ]:




