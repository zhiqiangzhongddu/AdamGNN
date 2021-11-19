import sys
from argparse import ArgumentParser
import numpy as np
import torch
import torch_geometric as tg

from load_dataset import get_LP_dataset, get_NC_dataset, get_GC_dataset
from set_up_training import set_up_LP_training, set_up_NC_training
from prepare_data import prepare_LP_data, prepare_NC_data
from utils import weights_init, seed_everything
from execution import NC_run, LP_run, GC_run
from model import AdamGNN, GAE, InnerProductDecoder

import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

# sys.argv = ['']  # execution on jupyter notebook
parser = ArgumentParser()
# general
parser.add_argument('--task', dest='task', default='LP', type=str,
                    help='LP; NC; GC')
parser.add_argument('--data_name', dest='data_name', default='emails', type=str,
                    help='cora; emails; ...')
parser.add_argument('--mode', dest='mode', default='baseline', type=str,
                    help='experiment mode. E.g., baseline or basemodel')
parser.add_argument('--model', dest='model', default='GCN', type=str,
                    help='model class name. E.g., GCN, PGNN, HCGNN...')
parser.add_argument('--local_agg_gnn', dest='local_agg_gnn', default='GCN', type=str,
                    help='GNN model used for primary node generation')
parser.add_argument('--fitness_mode', dest='fitness_mode', default='both_c', type=str,
                    help='how do we jointly use two fitness scores. E.g., c: \phi_c s: \phi_s both_j both_c')
parser.add_argument('--pooling_mode', dest='pooling_mode', default='att', type=str,
                    help='adaptive pooling mode. E.g., mean, max, att')
parser.add_argument('--num_levels', dest='num_levels', default=4, type=int,
                    help='number of hierarchical levels.')
parser.add_argument('--hid_dim', dest='hid_dim', default=64, type=int,
                    help='the hidden dimentin of neural network')
parser.add_argument('--cluster_range', dest='cluster_range', default=1, type=int,
                    help='number of hops to define the ego-network')
parser.add_argument('--overlap', dest='overlap', default=True, type=bool,
                    help='whether allow overlapping between different selected ego-networks')
parser.add_argument('--loss_mode', dest='loss_mode', default='all', type=str,
                    help='mode of loss fucntion. E.g., task, KL, R')
parser.add_argument('--batch_size', dest='batch_size', default=64, type=int,
                    help='batch_size')
parser.add_argument('--use_attribute', dest='use_attribute', default=True, type=bool,
                    help='wheter adopt attributes of dataset')
parser.add_argument('--output_mode', dest='output_mode', default='ATT', type=str,
                    help='the mode of flyback aggregator. E.g., ATT, MEAN, MAC, LSTM')
parser.add_argument('--gat_head', dest='gat_head', default=0, type=int,
                    help='the number of attention head if use GAT for primary node embedding generation')
parser.add_argument('--all_cluster', dest='all_cluster', default=True, type=bool,
                    help='whether pick all ego-networks that satisfy selection requirements')
parser.add_argument('--pooling_ratio', dest='pooling_ratio', default=1, type=int,
                    help='the ratio of selection if do not select all satisfied ego-networks')
parser.add_argument('--l2_regularization', dest='l2_regularization', default=5e-4, type=float,
                    help='l2 regularization value')
parser.add_argument('--edge_threshold', dest='edge_threshold', default=0, type=float,
                    help='filter weak edges')
parser.add_argument('--do_view', dest='do_view', default=False, type=bool,
                    help='whether present detailed training process')
parser.add_argument('--early_stop', dest='early_stop', default=100, type=int,
                    help='patience to wait for training')

parser.add_argument('--gpu', dest='gpu', default=True, type=bool,
                    help='whether use gpu')
parser.add_argument('--seed', dest='seed', default=123, type=int)

# dataset
parser.add_argument('--num_train', dest='num_train', default=80, type=float)
parser.add_argument('--use_features', dest='use_features', default=True, type=bool,
                    help='whether use node features')

# model
parser.add_argument('--lr', dest='lr', default=1e-2, type=float)
parser.add_argument('--num_epoch', dest='num_epoch', default=201, type=int)
parser.add_argument('--verbose', dest='verbose', default=1, type=int)
parser.add_argument('--relu', dest='relu', default=True, type=bool)
parser.add_argument('--dropout', dest='dropout', default=False, type=bool)
parser.add_argument('--drop_ratio', dest='drop_ratio', default=0.5, type=float)

args = parser.parse_args()

args.device = torch.device('cuda:' + str(0) if args.gpu and torch.cuda.is_available() else 'cpu')
seed_everything(args.seed)
print(args)
###########################################################################

if args.task == 'LP':
    # load dataset
    data = get_LP_dataset(
        dataset_name=args.data_name,
        use_features=args.use_attribute,
        seed=args.seed
    )

    # set up training mode
    graph, df_train, df_valid, df_test = set_up_LP_training(
        data=data, num_train=args.num_train, seed=args.seed
    )
    # prepare data
    data = prepare_LP_data(
        data=data,
        df_train=df_train, df_valid=df_valid, df_test=df_test,
        batch_size=args.batch_size, device=args.device
    )

    print('train {}, valid {}, test {}'.format(
        data.train_pos_edge_index.shape[1],
        data.val_pos_edge_index.shape[1],
        data.test_pos_edge_index.shape[1]
    ))

    encoder = AdamGNN(
        args=args, feat_dim=data.x.shape[1], out_dim=args.hid_dim
    )
    model = GAE(encoder=encoder, decoder=InnerProductDecoder())
    model = model.to(args.device)
    model.apply(weights_init)
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr, weight_decay=args.l2_regularization
    )

    LP_run(
        args=args, model=model, optimizer=optimizer, data=data
    )

elif args.task == 'NC':
    # load dataset
    data = get_NC_dataset(
        dataset_name=args.data_name, use_features=args.use_attribute, seed=args.seed
    )
    # set up training mode
    data = set_up_NC_training(
        data=data, num_train=args.num_train, seed=args.seed
    )
    # prepare data
    data = prepare_NC_data(
        data=data, batch_size=args.batch_size,
        dense=False, device=args.device, seed=args.seed
    )
    norm = tg.transforms.NormalizeFeatures()
    data = norm(data)
    # set up model
    model = AdamGNN(
        args=args, feat_dim=data.x.shape[1], out_dim=data.y.unique().size(0)
    )
    # release gpu memory
    torch.cuda.empty_cache()
    model = model.to(args.device)
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr, weight_decay=args.l2_regularization
    )
    NC_run(
        args=args, model=model, optimizer=optimizer, data=data
    )


elif args.task == 'GC':
    # load dataset
    train_loader, val_loader, test_loader = get_GC_dataset(
        dataset_name=args.data_name,
        batch_size=args.batch_size, local_test=args.local_test, seed=args.seed
    )

    if train_loader.dataset[0].x is None:
        model = AdamGNN(
            args=args, feat_dim=128,
            out_dim=np.unique(
                np.concatenate([data.y.unique().data.cpu().numpy() for data in train_loader.dataset])
            ).shape[0]
        )
    else:
        model = AdamGNN(
            args=args, feat_dim=train_loader.dataset[0].x.shape[1],
            out_dim=np.unique(
                np.concatenate([data.y.unique().data.cpu().numpy() for data in train_loader.dataset])
            ).shape[0]
        )

    # release gpu memory
    torch.cuda.empty_cache()

    model = model.to(args.device)
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr,
        weight_decay=args.l2_regularization
    )
    GC_run(
        args=args, model=model, optimizer=optimizer,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader
    )
