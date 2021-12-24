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


def set_up_NC_training(data, num_train, seed):
    random.seed(seed)
    np.random.seed(seed)

    if num_train != 20:
        if num_train < 10:
            '''
            few-shot settings
            In this case, num_train is number of training instances, instead of train ratio. 
            '''
            train_nodes = np.empty(0, dtype=int)
            val_nodes = np.empty(0, dtype=int)
            for label in data.y.unique():
                index = torch.nonzero(data.y == label).view(-1)
                select_train = np.array(random.sample(index.tolist(), num_train))
                select_val = random.sample(list(set(index.tolist()) - set(select_train)), num_train)
                train_nodes = np.append(train_nodes, select_train)
                val_nodes = np.append(val_nodes, select_val)
            test_nodes = np.array(list(set(range(data.num_nodes)) - set(train_nodes) - set(val_nodes)))
        else:
            non_train_nodes = random.sample(
                list(range(data.num_nodes)), int(data.num_nodes * ((1 - num_train / 100)))
            )
            train_nodes = np.array(list(set(range(data.num_nodes)) - set(non_train_nodes)))
            val_nodes = np.array(non_train_nodes[:int(data.num_nodes * ((1 - num_train / 100)) / 2)])
            test_nodes = np.array(non_train_nodes[int(data.num_nodes * ((1 - num_train / 100)) / 2):])

        train_mask = np.array([False] * data.num_nodes)
        val_mask = np.array([False] * data.num_nodes)
        test_mask = np.array([False] * data.num_nodes)
        train_mask[train_nodes] = True
        val_mask[val_nodes] = True
        test_mask[test_nodes] = True

        data.train_mask = torch.Tensor(train_mask).bool()
        data.val_mask = torch.Tensor(val_mask).bool()
        data.test_mask = torch.Tensor(test_mask).bool()

    print('Train: {} valid: {} test: {}'.format(
        data.train_mask.sum().numpy().tolist(),
        data.val_mask.sum().numpy().tolist(),
        data.test_mask.sum().numpy().tolist()
    ))

    return data


def set_up_LP_training_pyg(data, ratio_val, ratio_test, seed):
    # data.train_mask = data.val_mask = data.test_mask = data.y = None
    data.train_mask = data.val_mask = data.test_mask = None

    data = train_test_split_edges(data=data, val_ratio=ratio_val, test_ratio=ratio_test, seed=seed)
    print(f'Valid: {data.val_pos_edge_index.shape[1]}, test: {data.test_pos_edge_index.shape[1]}')

    return data


def set_up_LP_training(data, num_train, seed):
    random.seed(seed)
    np.random.seed(seed)

    data.edge_index = tg.utils.remove_self_loops(edge_index=data.edge_index)[0]
    graph = tg.utils.to_networkx(data).to_undirected()
    graph_complete = deepcopy(graph)

    # collect negative No-train edges
    print(f'Graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.')
    start = time.time()
    print('we need to collect {} negative edges.'.format(
        int(graph.number_of_edges() * ((100 - num_train) / 100))
    ))

    df_train_edges_pos = nx.to_pandas_edgelist(graph)[['source', 'target']]
    df_no_train_edges_neg = pd.DataFrame(
        np.random.choice(list(graph.nodes()), 10 * graph.number_of_edges()), columns=['source']
    )
    df_no_train_edges_neg['target'] = np.random.choice(list(graph.nodes()), 10 * graph.number_of_edges())
    df_no_train_edges_neg = df_no_train_edges_neg[df_no_train_edges_neg['source'] < df_no_train_edges_neg['target']]
    df_no_train_edges_neg = df_no_train_edges_neg.drop_duplicates().reset_index(drop=True)
    df_no_train_edges_neg = pd.merge(
        df_train_edges_pos, df_no_train_edges_neg, indicator=True, how='outer'
    ).query('_merge=="right_only"').drop('_merge', axis=1).reset_index(drop=True)
    no_train_edges_neg = random.sample(
        df_no_train_edges_neg.values.tolist(), int(graph_complete.number_of_edges() * ((100 - num_train) / 100))
    )
    print('Generating negative instances uses {:.2f} seconds.'.format(time.time() - start))

    # collect positive No-train edges
    start = time.time()
    print('We need to remove {} edges.'.format(int(graph_complete.number_of_edges() * ((100 - num_train) / 100))))
    # df_train_edges_pos = nx.to_pandas_edgelist(graph_complete)
    G_train = nx.Graph(graph_complete)
    edge_index = np.array(list(graph_complete.edges))
    edges = np.transpose(edge_index)

    e = edges.shape[1]
    edges = edges[:, np.random.permutation(e)]
    unique, counts = np.unique(edges, return_counts=True)
    node_count = dict(zip(unique, counts))

    index_train = []
    index_val = []
    for i in range(e):
        node1 = edges[0, i]
        node2 = edges[1, i]
        if node_count[node1] > 1 and node_count[node2] > 1:  # if degree>1
            index_val.append(i)
            node_count[node1] -= 1
            node_count[node2] -= 1
            if len(index_val) == int(e * (100 - num_train) / 100):
                break
        else:
            index_train.append(i)
    # index_train = index_train + list(range(i + 1, e))

    # edges_train = edges[:, index_train]
    edges_no_train = edges[:, index_val]
    no_train_edges_pos = [[edges_no_train[0, i], edges_no_train[1, i]] for i in range(edges_no_train.shape[1])]

    G_train.remove_edges_from(no_train_edges_pos)
    if len(no_train_edges_pos) < int(graph_complete.number_of_edges() * ((100 - num_train) / 100)):
        print('There are only {} positive instances.'.format(len(no_train_edges_pos)))
        sys.exit("Can not remove more edges.")
    print('Generating positive instances uses {:.2f} seconds.'.format(time.time() - start))
    graph = G_train

    df_friends = nx.to_pandas_edgelist(graph)[['source', 'target']]
    # _x = deepcopy(df_friends)
    # _x.columns = ['target', 'source']
    # df_friends = pd.concat([df_friends, _x]).reset_index(drop=True)

    valid_edges_pos = random.sample(no_train_edges_pos,
                                    int(graph_complete.number_of_edges() * ((100 - num_train) / 100) / 2))
    valid_edges_neg = random.sample(no_train_edges_neg,
                                    int(graph_complete.number_of_edges() * ((100 - num_train) / 100) / 2))
    test_edges_pos = [item for item in no_train_edges_pos if item not in valid_edges_pos]
    test_edges_neg = [item for item in no_train_edges_neg if item not in valid_edges_neg]

    test_edges = {
        'positive': test_edges_pos,
        'negative': test_edges_neg
    }
    valid_edges = {
        'positive': valid_edges_pos,
        'negative': valid_edges_neg
    }

    # valid data
    df_valid_pos_samples = pd.DataFrame(valid_edges['positive'], columns=['source', 'target'])
    df_valid_pos_samples['label'] = 1
    df_valid_neg_samples = pd.DataFrame(valid_edges['negative'], columns=['source', 'target'])
    df_valid_neg_samples['label'] = 0
    df_valid = pd.concat([df_valid_pos_samples, df_valid_neg_samples], axis=0)

    # test data
    df_test_pos_samples = pd.DataFrame(test_edges['positive'], columns=['source', 'target'])
    df_test_pos_samples['label'] = 1
    df_test_neg_samples = pd.DataFrame(test_edges['negative'], columns=['source', 'target'])
    df_test_neg_samples['label'] = 0
    df_test = pd.concat([df_test_pos_samples, df_test_neg_samples], axis=0)

    # train data
    df_train_neg = pd.DataFrame(
        np.random.choice(list(graph.nodes()), 10 * graph.number_of_edges()), columns=['source']
    )
    df_train_neg['target'] = np.random.choice(list(graph.nodes()), 10 * graph.number_of_edges())
    df_train_neg = df_train_neg[df_train_neg['source'] < df_train_neg['target']]
    df_train_neg = df_train_neg.drop_duplicates().reset_index(drop=True)
    # df_train_pos = df_friends[df_friends['source'] < df_friends['target']]
    df_train_pos = df_friends
    df_train_pos['label'] = 1
    df_non = pd.concat([df_train_pos, df_valid, df_test]).reset_index(drop=True)[['source', 'target']]
    df_train_neg = pd.merge(
        df_non, df_train_neg, indicator=True, how='outer'
    ).query('_merge=="right_only"').drop('_merge', axis=1).reset_index(drop=True)
    df_train_neg = df_train_neg.sample(df_train_pos.shape[0], random_state=seed)
    df_train_neg['label'] = 0
    df_train = pd.concat([df_train_pos, df_train_neg]).drop_duplicates().reset_index(drop=True)

    return graph, df_train, df_valid, df_test
