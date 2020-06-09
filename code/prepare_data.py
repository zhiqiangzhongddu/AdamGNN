import numpy as np
import pandas as pd

import torch
import torch_geometric as tg
import random
from utils import *


def prepare_NC_data(ls_data, batch_size, device, SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    res = []
    for idx, data in enumerate(ls_data):
        # data.edge_index = tg.utils.add_self_loops(data.edge_index)[0]

        edge_matrix = tg.utils.to_dense_adj(data.edge_index, batch=None)[0]
        
        if batch_size is not None:
            num_batch = int(np.ceil(data.num_nodes / batch_size))
        else:
            num_batch = 1
        batch = torch.cat([torch.FloatTensor([n]*int(np.ceil(data.num_nodes/num_batch)+1))\
                        for n in range(num_batch)]).long()[:data.num_nodes]
        
        data.edge_matrix = edge_matrix
        data.batch = batch
        data.to(device)
        # ls_data[idx] = data
        res.append(data)
        
        # release gpu memory
        del data, edge_matrix
        torch.cuda.empty_cache()

    return res

def prepare_LP_data_pyg(ls_data, batch_size, device):
    for idx, data in enumerate(ls_data):
        edge_matrix = tg.utils.to_dense_adj(data.train_pos_edge_index, batch=None)[0].to(device)
        
        if batch_size is not None:
            num_batch = int(np.ceil(data.num_nodes / batch_size))
        else:
            num_batch = 1
        batch = torch.cat([torch.FloatTensor([n]*int(np.ceil(data.num_nodes/num_batch)+1))\
                        for n in range(num_batch)])[:data.num_nodes]
        data.edge_matrix = edge_matrix
        data.batch = batch
        data.train_neg_adj_mask = None

        ls_data[idx] = data.to(device)
    return ls_data


def prepare_LP_data(ls_data, ls_df_train, ls_df_valid, ls_df_test, batch_size, device, SEED):

    for idx, data in enumerate(ls_data):
        data.train_mask = data.val_mask = data.test_mask = None
        
        df_train_pos = ls_df_train[idx][ls_df_train[idx]['label']==1]
        df_train_neg = ls_df_train[idx][ls_df_train[idx]['label']==0]
        df_val_pos = ls_df_valid[idx][ls_df_valid[idx]['label']==1]
        df_val_neg = ls_df_valid[idx][ls_df_valid[idx]['label']==0]
        df_test_pos = ls_df_test[idx][ls_df_test[idx]['label']==1]
        df_test_neg = ls_df_test[idx][ls_df_test[idx]['label']==0]
        
        train_pos_edge_index = torch.tensor(df_train_pos[['source', 'target']].values).t()
        train_neg_edge_index = torch.tensor(df_train_neg[['source', 'target']].values).t()
        val_pos_edge_index = torch.tensor(df_val_pos[['source', 'target']].values).t()
        val_neg_edge_index   = torch.tensor(df_val_neg[['source', 'target']].values).t()
        test_pos_edge_index = torch.tensor(df_test_pos[['source', 'target']].values).t()
        test_neg_edge_index = torch.tensor(df_test_neg[['source', 'target']].values).t()
        
        if batch_size is not None:
            num_batch = int(np.ceil(data.num_nodes / batch_size))
        else:
            num_batch = 1
        batch = torch.cat([torch.FloatTensor([n]*int(np.ceil(data.num_nodes/num_batch)+1))\
                        for n in range(num_batch)])[:data.num_nodes]
        
        data.train_pos_edge_index = train_pos_edge_index
        data.train_neg_edge_index = train_neg_edge_index
        data.val_pos_edge_index = val_pos_edge_index
        data.val_neg_edge_index = val_neg_edge_index
        data.test_pos_edge_index = test_pos_edge_index
        data.test_neg_edge_index = test_neg_edge_index
        
        data.train_pos_edge_index = tg.utils.to_undirected(data.train_pos_edge_index)
        data.train_pos_edge_index = tg.utils.remove_self_loops(data.train_pos_edge_index)[0]
        # data.train_pos_edge_index = tg.utils.add_self_loops(data.train_pos_edge_index)[0]

        data.batch = batch
        edge_matrix = tg.utils.to_dense_adj(data.train_pos_edge_index, batch=None)[0].to(device)
        data.edge_matrix = edge_matrix
        data.edge_index = None

        ls_data[idx] = data.to(device)
        # release gpu memory
        del data, edge_matrix
        torch.cuda.empty_cache()
    return ls_data