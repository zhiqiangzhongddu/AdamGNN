import numpy as np
import pandas as pd
import random
import time
import networkx as nx
from copy import deepcopy
import sys
import torch
import torch_geometric as tg
from utils import train_test_split_edges

def set_up_NC_training_dphi_2(ls_data, num_train, SEED):
    random.seed(SEED)
    np.random.seed(SEED)

    for idx, data in enumerate(ls_data):
        available_nodes = data.available_nodes
        num_available_node = len(available_nodes)

        non_train_nodes = random.sample(list(range(num_available_node)), int(num_available_node*((1-num_train/100))))
        train_nodes = np.array(list(set(range(num_available_node))-set(non_train_nodes)))
        val_nodes = np.array(non_train_nodes[:int(num_available_node*((1-num_train/100))/2)])
        test_nodes = np.array(non_train_nodes[int(num_available_node*((1-num_train/100))/2):])

        train_nodes = np.array(available_nodes)[train_nodes]
        val_nodes = np.array(available_nodes)[val_nodes]
        test_nodes = np.array(available_nodes)[test_nodes]

        train_mask = np.array([False] * data.num_nodes)
        val_mask = np.array([False] * data.num_nodes)
        test_mask = np.array([False] * data.num_nodes)
        train_mask[train_nodes] = True
        val_mask[val_nodes] = True
        test_mask[test_nodes] = True

        data.train_mask = torch.Tensor(train_mask).bool()
        data.val_mask = torch.Tensor(val_mask).bool()
        data.test_mask = torch.Tensor(test_mask).bool()

        print(f'For graph {idx}, train: {data.train_mask.sum().numpy().tolist()} valid: {data.val_mask.sum().numpy().tolist()} test: {data.test_mask.sum().numpy().tolist()}')
        ls_data[idx] = data
    return ls_data

def set_up_NC_training(ls_data, num_train, SEED):
    random.seed(SEED)
    np.random.seed(SEED)

    for idx, data in enumerate(ls_data):
        if data.num_nodes < 2000:
            if num_train != 20:
                non_train_nodes = random.sample(list(range(data.num_nodes)), int(data.num_nodes*((1-num_train/100))))
                train_nodes = np.array(list(set(range(data.num_nodes))-set(non_train_nodes)))
                val_nodes = np.array(non_train_nodes[:int(data.num_nodes*((1-num_train/100))/2)])
                test_nodes = np.array(non_train_nodes[int(data.num_nodes*((1-num_train/100))/2):])

                train_mask = np.array([False] * data.num_nodes)
                val_mask = np.array([False] * data.num_nodes)
                test_mask = np.array([False] * data.num_nodes)
                train_mask[train_nodes] = True
                val_mask[val_nodes] = True
                test_mask[test_nodes] = True

                data.train_mask = torch.Tensor(train_mask).bool()
                data.val_mask = torch.Tensor(val_mask).bool()
                data.test_mask = torch.Tensor(test_mask).bool()
        else:
            if num_train != 20:
                if num_train > 50:
                    non_train_nodes = random.sample(list(range(data.num_nodes)), int(data.num_nodes*((1-num_train/100))))
                    train_nodes = np.array(list(set(range(data.num_nodes))-set(non_train_nodes)))
                    val_nodes = np.array(non_train_nodes[:int(data.num_nodes*((1-num_train/100))/2)])
                    test_nodes = np.array(non_train_nodes[int(data.num_nodes*((1-num_train/100))/2):])

                    train_mask = np.array([False] * data.num_nodes)
                    val_mask = np.array([False] * data.num_nodes)
                    test_mask = np.array([False] * data.num_nodes)
                    train_mask[train_nodes] = True
                    val_mask[val_nodes] = True
                    test_mask[test_nodes] = True

                    data.train_mask = torch.Tensor(train_mask).bool()
                    data.val_mask = torch.Tensor(val_mask).bool()
                    data.test_mask = torch.Tensor(test_mask).bool()
                else:
                    non_train_nodes = random.sample(list(range(data.num_nodes)), int(data.num_nodes*((1-num_train/100))))
                    train_nodes = np.array(list(set(range(data.num_nodes))-set(non_train_nodes)))
                    val_nodes = np.array(random.sample(set(range(data.num_nodes))-set(train_nodes), 500))
                    test_nodes = np.array(random.sample(set(range(data.num_nodes))-set(train_nodes)-set(val_nodes), 1000))

                    train_mask = np.array([False] * data.num_nodes)
                    val_mask = np.array([False] * data.num_nodes)
                    test_mask = np.array([False] * data.num_nodes)
                    train_mask[train_nodes] = True
                    val_mask[val_nodes] = True
                    test_mask[test_nodes] = True

                    data.train_mask = torch.Tensor(train_mask).bool()
                    data.val_mask = torch.Tensor(val_mask).bool()
                    data.test_mask = torch.Tensor(test_mask).bool()

        print(f'For graph {idx}, train: {data.train_mask.sum().numpy().tolist()} valid: {data.val_mask.sum().numpy().tolist()} test: {data.test_mask.sum().numpy().tolist()}')
        ls_data[idx] = data
    return ls_data


def set_up_LP_training_pyg(ls_data, ratio_val, ratio_test, SEED):
    for idx, data in enumerate(ls_data):
        # data.train_mask = data.val_mask = data.test_mask = data.y = None
        data.train_mask = data.val_mask = data.test_mask = None
        
        data = train_test_split_edges(data=data, val_ratio=ratio_val, test_ratio=ratio_test, SEED=SEED)
        print(f'For graph {idx}, valid: {data.val_pos_edge_index.shape[1]}, test: {data.test_pos_edge_index.shape[1]}')
        ls_data[idx] = data
    return ls_data


def set_up_LP_training(ls_data, num_train, SEED):
    random.seed(SEED)
    np.random.seed(SEED)

    graphs = [tg.utils.to_networkx(data).to_undirected() for data in ls_data]
    graphs_all = deepcopy(graphs)

    # collect negative No-train edges
    ls_no_train_edges_neg = []
    for idx, graph in enumerate(graphs):
        #
        graph.remove_edges_from(nx.selfloop_edges(graph))
        
        print(f'Graph {idx} has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.')
        start = time.time()
        print('For graph {}, we need to collect {} negative edges.'.\
            format(idx, int(graph.number_of_edges()*((100-num_train)/100))))

        df_train_edges_pos = nx.to_pandas_edgelist(graph)[['source', 'target']]
        df_no_train_edges_neg = pd.DataFrame(np.random.choice(list(graph.nodes()), 10*graph.number_of_edges()), columns=['source'])
        df_no_train_edges_neg['target'] = np.random.choice(list(graph.nodes()), 10*graph.number_of_edges())
        df_no_train_edges_neg = df_no_train_edges_neg[df_no_train_edges_neg['source']<df_no_train_edges_neg['target']]
        df_no_train_edges_neg = df_no_train_edges_neg.drop_duplicates().reset_index(drop=True)
        df_no_train_edges_neg = pd.merge(df_train_edges_pos, df_no_train_edges_neg, indicator=True, how='outer').\
                                                query('_merge=="right_only"').\
                                                drop('_merge', axis=1).reset_index(drop=True)
        no_train_edges_neg = random.sample(df_no_train_edges_neg.values.tolist(), int(graphs_all[idx].number_of_edges() * ((100-num_train)/100)))
        print('Generating {} negative instances uses {:.2f} seconds.'.format(idx, time.time()-start))
        ls_no_train_edges_neg.append(no_train_edges_neg)
    
    # collect positive No-train edges
    ls_no_train_edges_pos = []
    for idx, graph in enumerate(graphs_all):
        start = time.time()
        #
        graph.remove_edges_from(nx.selfloop_edges(graph))
        
        print('For graph {}, we need to remove {} edges.'.format(idx, int(graph.number_of_edges()*((100-num_train)/100))))
        df_train_edges_pos = nx.to_pandas_edgelist(graph)
        G_train = nx.Graph(graph)
        edge_index = np.array(list(graph.edges))
        edges = np.transpose(edge_index)

        e = edges.shape[1]
        edges = edges[:, np.random.permutation(e)]
        unique, counts = np.unique(edges, return_counts=True)
        node_count = dict(zip(unique, counts))

        index_train = []
        index_val = []
        for i in range(e):
            node1 = edges[0,i]
            node2 = edges[1,i]
            if node_count[node1]>1 and node_count[node2]>1: # if degree>1
                index_val.append(i)
                node_count[node1] -= 1
                node_count[node2] -= 1
                if len(index_val) == int(e*(100-num_train)/100):
                    break
            else:
                index_train.append(i)
        index_train = index_train + list(range(i + 1, e))

        edges_train = edges[:, index_train]
        edges_no_train = edges[:, index_val]
        no_train_edges_pos = [[edges_no_train[0, i], edges_no_train[1, i]] for i in range(edges_no_train.shape[1])]

        G_train.remove_edges_from(no_train_edges_pos)
        if len(no_train_edges_pos) < int(graph.number_of_edges()*((100-num_train)/100)):
            print('For graph {}, there are only {} positive instances.'.format(idx, len(no_train_edges_pos)))
            sys.exit("Can not remove more edges.")
        print('Generating {} positive instances uses {:.2f} seconds.'.format(idx, time.time()-start))

        graphs[idx] = G_train
        ls_no_train_edges_pos.append(no_train_edges_pos)

    ls_df_friends = []
    for idx in range(len(graphs)):
        df_friends = nx.to_pandas_edgelist(graphs[idx])[['source', 'target']]

        _x = deepcopy(df_friends)
        _x.columns = ['target', 'source']
        df_friends = pd.concat([df_friends, _x]).reset_index(drop=True)

        ls_df_friends.append(df_friends)

    ls_valid_edges = []
    ls_test_edges = []
    for idx in range(len(graphs)):
        valid_edges_pos = random.sample(ls_no_train_edges_pos[idx], int(graphs_all[idx].number_of_edges()*((100-num_train)/100)/2))
        valid_edges_neg = random.sample(ls_no_train_edges_neg[idx], int(graphs_all[idx].number_of_edges()*((100-num_train)/100)/2))
        test_edges_pos = [item for item in ls_no_train_edges_pos[idx] if item not in valid_edges_pos]
        test_edges_neg = [item for item in ls_no_train_edges_neg[idx] if item not in valid_edges_neg]

        test_edges = {'positive': test_edges_pos,
                    'negative': test_edges_neg}
        valid_edges = {'positive': valid_edges_pos,
                    'negative': valid_edges_neg}
        ls_valid_edges.append(valid_edges)
        ls_test_edges.append(test_edges)

    # valid data
    ls_df_valid = []
    for idx, valid_edges in enumerate(ls_valid_edges):
        df_valid_pos_samples = pd.DataFrame(valid_edges['positive'], columns=['source', 'target'])
        df_valid_pos_samples['label'] = 1
        df_valid_neg_samples = pd.DataFrame(valid_edges['negative'], columns=['source', 'target'])
        df_valid_neg_samples['label'] = 0
        df_valid = pd.concat([df_valid_pos_samples, df_valid_neg_samples], axis=0) 
        ls_df_valid.append(df_valid)

    # test data
    ls_df_test = []
    for idx, test_edges in enumerate(ls_test_edges):
        df_test_pos_samples = pd.DataFrame(test_edges['positive'], columns=['source', 'target'])
        df_test_pos_samples['label'] = 1
        df_test_neg_samples = pd.DataFrame(test_edges['negative'], columns=['source', 'target'])
        df_test_neg_samples['label'] = 0
        df_test = pd.concat([df_test_pos_samples, df_test_neg_samples], axis=0)
        ls_df_test.append(df_test)

    # train data
    ls_df_train = []
    for idx, friends in enumerate(ls_df_friends):
        graph = graphs[idx]
        df_train_neg = pd.DataFrame(np.random.choice(list(graph.nodes()), 10*graph.number_of_edges()), columns=['source'])
        df_train_neg['target'] = np.random.choice(list(graph.nodes()), 10*graph.number_of_edges())
        df_train_neg = df_train_neg[df_train_neg['source']<df_train_neg['target']]
        df_train_neg = df_train_neg.drop_duplicates().reset_index(drop=True)

        df_valid = ls_df_valid[idx]
        df_test = ls_df_test[idx]
        df_train_pos = ls_df_friends[idx]
        df_train_pos = friends[friends['source']<friends['target']]
        df_train_pos['label'] = 1
        df_non = pd.concat([df_train_pos, df_valid, df_test]).reset_index(drop=True)[['source', 'target']]

        df_train_neg = pd.merge(df_non, df_train_neg, indicator=True, how='outer').\
                                    query('_merge=="right_only"').\
                                    drop('_merge', axis=1).reset_index(drop=True)
        df_train_neg = df_train_neg.sample(df_train_pos.shape[0], random_state=SEED)  
        df_train_neg['label'] = 0
        df_train = pd.concat([df_train_pos, df_train_neg]).drop_duplicates().reset_index(drop=True)
        ls_df_train.append(df_train)

    return graphs, ls_df_train, ls_df_valid, ls_df_test

