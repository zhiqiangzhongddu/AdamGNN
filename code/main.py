import pandas as pd
import numpy as np
import math
import random
import networkx as nx
import time
from copy import deepcopy
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, normalized_mutual_info_score
import sys
import os

import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import torch_geometric as tg
from torch_geometric.nn import GraphConv, GCNConv, SAGEConv, GATConv, GINConv, JumpingKnowledge, GAE, InnerProductDecoder
from torch_geometric.nn.pool.topk_pool import topk

from load_dataset import *
from set_up_training import *
from prepare_data import *
from utils import *
from run import *
from args import *
from model import *

args = make_args()
model_config = vars(args)
print(args)
# SEED = random.randint(1,10001)
SEED = args.SEED
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline
###########################################################################

if not os.path.isdir('runs'):
    os.mkdir('runs')

device = torch.device('cuda:'+str(0) if args.gpu and torch.cuda.is_available() else 'cpu')
model_config['device'] = device

loc_time = time.localtime()
model_config['model_dir'] = 'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
model_config['alias'] = 'test_model_{}_{}_{}_{}_{}'.format(loc_time[0], loc_time[1], loc_time[2], loc_time[3], loc_time[4])
###########################################################################

def weights_init(m):
        if isinstance(m, torch.nn.Linear):
            m.weight.data = torch.nn.init.uniform_(m.weight.data)
            # m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
            # m.weight.data = torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            # pass

if args.task == 'LP':
    print(0)
    # load dataset
    ls_data = get_LP_dataset(dataset_name=args.data_name,
                            use_features=args.use_attribute,
                            SEED=args.SEED)

    # set up training mode
    graphs, ls_df_train, ls_df_valid, ls_df_test = set_up_LP_training(ls_data=ls_data,
                                                                    num_train=args.ratio_train, 
                                                                    SEED=args.SEED)
    # prepare data
    ls_data = prepare_LP_data(ls_data=ls_data,
                            ls_df_train=ls_df_train, 
                            ls_df_valid=ls_df_valid, 
                            ls_df_test=ls_df_test, 
                            batch_size=args.batch_size,
                            device=device,
                            SEED=args.SEED)

    print(f'train {ls_data[0].train_pos_edge_index.shape[1]}, valid {ls_data[0].val_pos_edge_index.shape[1]}, test {ls_data[0].test_pos_edge_index.shape[1]}')

    encoder = AHGNN_LP(config=model_config,
                        feat_dim=ls_data[0].x.shape[1])
    model = GAE(encoder=encoder, decoder=InnerProductDecoder())
    model = model.to(device)
    model.apply(weights_init)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=args.lr,
                                weight_decay=args.l2_regularization)

    LP_run(model=model, 
        optimizer=optimizer, 
        ls_data=ls_data, 
        model_config=model_config, 
        n_clusters=ls_data[0].y.unique().shape[0],
        device=device, 
        SEED=args.SEED)

elif args.task == 'NC':
    # load dataset
    ls_data = get_NC_dataset(dataset_name=args.data_name,
                            use_features=args.use_attribute,
                            SEED=args.SEED)
    # set up training mode
    ls_data = set_up_NC_training(ls_data=ls_data, 
                                num_train=args.num_train, 
                                SEED=args.SEED)
    # prepare data
    ls_data = prepare_NC_data(ls_data=ls_data, 
                            batch_size=args.batch_size, 
                            device=device, 
                            SEED=args.SEED)

    model = AHGNN_NC(config=model_config,
                feat_dim=ls_data[0].x.shape[1],
                out_dim=ls_data[0].y.unique().size(0))

    # release gpu memory
    torch.cuda.empty_cache()

    model = model.to(device)
    model.apply(weights_init)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=args.lr,
                                weight_decay=args.l2_regularization)


elif args.task == 'GC':
    # load dataset
    train_loader, val_loader, test_loader = get_GC_dataset(dataset_name=args.data_name,
                                                        use_features=args.use_attribute,
                                                        batch_size=args.batch_size,
                                                        local_test=args.local_test,
                                                        SEED=args.SEED)

    if train_loader.dataset[0].x is None:
        model = AHGNN_GC(config=model_config,
                feat_dim=128,
                out_dim=np.unique(np.concatenate([data.y.unique().data.cpu().numpy() for data in train_loader.dataset])).shape[0])
    else:
        model = AHGNN_GC(config=model_config,
                    feat_dim=train_loader.dataset[0].x.shape[1],
                    out_dim=np.unique(np.concatenate([data.y.unique().data.cpu().numpy() for data in train_loader.dataset])).shape[0])

    # release gpu memory
    torch.cuda.empty_cache()

    model = model.to(device)
    model.apply(weights_init)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=args.lr,
                                weight_decay=args.l2_regularization)
    