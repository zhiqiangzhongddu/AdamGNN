from argparse import ArgumentParser
def make_args():
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
    parser.add_argument('--SEED', dest='SEED', default=123, type=int)
    
    # dataset
    parser.add_argument('--ratio_train', dest='ratio_train', default=80, type=float)
    parser.add_argument('--use_features', dest='use_features', default=True, type=bool,
                        help='whether use node features')
    
    # model
    parser.add_argument('--lr', dest='lr', default=1e-2, type=float)
    parser.add_argument('--num_epoch', dest='num_epoch', default=201, type=int)
    parser.add_argument('--verbose', dest='verbose', default=1, type=int)
    parser.add_argument('--relu', dest='relu', default=True, type=bool)
    parser.add_argument('--dropout', dest='dropout', default=False, type=bool)
    parser.add_argument('--drop_ratio', dest='drop_ratio', default=0.5, type=float)
    
    parser.set_defaults(gpu=True, task='LP', model='GCN', dataset='grid')
    args = parser.parse_args()
    return args