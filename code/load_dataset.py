import numpy as np
import scipy.sparse as sp
import pandas as pd
import networkx as nx
import random
from sklearn import preprocessing

import torch_geometric as tg
import torch_geometric.transforms as T
import torch


def get_GC_dataset(dataset_name, batch_size, local_test, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = tg.datasets.TUDataset('../input/', name=dataset_name, use_node_attr=True)
    dataset = dataset.shuffle()

    if local_test:
        test_dataset = dataset[:20]
        val_dataset = dataset[2:2 * 20]
        train_dataset = dataset[2 * 20: 200]
    else:
        n = (len(dataset) + 9) // 10
        test_dataset = dataset[:n]
        val_dataset = dataset[n:2 * n]
        train_dataset = dataset[2 * n:]

    test_loader = tg.data.DataLoader(test_dataset, batch_size=batch_size)
    val_loader = tg.data.DataLoader(val_dataset, batch_size=batch_size)
    train_loader = tg.data.DataLoader(train_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def get_NC_dataset(dataset_name, use_features, seed):
    random.seed(seed)
    np.random.seed(seed)

    if dataset_name == 'emails':
        print('is reading {} dataset...'.format(dataset_name))
        df = pd.read_csv('../data/emails/email.txt', header=None, sep=' ', names=['source', 'target'])
        graph = nx.from_pandas_edgelist(df=df, source='source', target='target', edge_attr=None)

        df_label = pd.read_csv('../data/emails/email_labels.txt', header=None, sep=' ', names=['node_id', 'label'])
        df_label = df_label[
            df_label['label'].isin(df_label['label'].value_counts()[df_label['label'].value_counts() > 20].index)]
        available_nodes = df_label['node_id'].unique()

        graph = graph.subgraph(available_nodes)
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))

        graph = nx.relabel_nodes(graph, mapping, copy=True)
        df_label['node_id'] = df_label['node_id'].replace(mapping)
        df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

        # ecode label into numeric and set them in order
        le = preprocessing.LabelEncoder()
        df_label['label'] = le.fit_transform(df_label['label'])

        identify_oh_feature = torch.FloatTensor(np.identity(graph.number_of_nodes()))

        data = tg.data.Data(
            x=identify_oh_feature,
            y=torch.LongTensor(df_label['label']),
            edge_index=tg.utils.from_networkx(G=graph).edge_index
        )

        # non_train_nodes = random.sample(list(range(data.num_nodes)), int(data.num_nodes*((1-20/100))))
        # train_nodes = np.array(list(set(range(data.num_nodes))-set(non_train_nodes)))
        train_nodes = np.empty(0, dtype=int)
        for label in data.y.unique():
            index = torch.nonzero(data.y == label).view(-1)
            train_nodes = np.append(train_nodes, np.array(random.sample(index.data.numpy().tolist(), 10)))
        non_train_nodes = list(set(np.arange(data.num_nodes)) - set(train_nodes))
        val_nodes = np.array(random.sample(non_train_nodes, len(non_train_nodes) // 2))
        test_nodes = np.array(list(set(non_train_nodes) - set(val_nodes)))

    elif dataset_name == 'cora':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Planetoid(
            root='../input/', name='Cora', transform=T.NormalizeFeatures(), pre_transform=None
        )[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.FloatTensor(feature)

    elif dataset_name == 'citeseer':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Planetoid(
            root='../input/', name='CiteSeer', transform=T.NormalizeFeatures(), pre_transform=None
        )[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.FloatTensor(feature)

    elif dataset_name == 'pubmed':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Planetoid(
            root='../input/', name='PubMed', transform=T.NormalizeFeatures(), pre_transform=None
        )[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.FloatTensor(feature)

    elif dataset_name == 'cs':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Coauthor(root='../input/', name='CS', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.FloatTensor(feature)

        train_nodes = np.empty(0, dtype=int)
        for label in data.y.unique():
            index = torch.nonzero(data.y == label).view(-1)
            train_nodes = np.append(train_nodes, np.array(random.sample(index.data.numpy().tolist(), 20)))
        val_nodes = np.array(random.sample(set(range(data.num_nodes)) - set(train_nodes), 500))
        test_nodes = np.array(random.sample(set(range(data.num_nodes)) - set(train_nodes) - set(val_nodes), 1000))

    elif dataset_name == 'physic':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Coauthor(root='../input/', name='Physics', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.FloatTensor(feature)

        train_nodes = np.empty(0, dtype=int)
        for label in data.y.unique():
            index = torch.nonzero(data.y == label).view(-1)
            train_nodes = np.append(train_nodes, np.array(random.sample(index.data.numpy().tolist(), 20)))
        val_nodes = np.array(random.sample(set(range(data.num_nodes)) - set(train_nodes), 500))
        test_nodes = np.array(random.sample(set(range(data.num_nodes)) - set(train_nodes) - set(val_nodes), 1000))

    elif dataset_name == 'computers':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Amazon(root='../input/', name='Computers', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.FloatTensor(feature)

        train_nodes = np.empty(0, dtype=int)
        for label in data.y.unique():
            index = torch.nonzero(data.y == label).view(-1)
            train_nodes = np.append(train_nodes, np.array(random.sample(index.data.numpy().tolist(), 20)))
        val_nodes = np.array(random.sample(set(range(data.num_nodes)) - set(train_nodes), 500))
        test_nodes = np.array(random.sample(set(range(data.num_nodes)) - set(train_nodes) - set(val_nodes), 1000))

    elif dataset_name == 'photo':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Amazon(root='../input/', name='Photo', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.FloatTensor(feature)

        train_nodes = np.empty(0, dtype=int)
        for label in data.y.unique():
            index = torch.nonzero(data.y == label).view(-1)
            train_nodes = np.append(train_nodes, np.array(random.sample(index.data.numpy().tolist(), 20)))
        val_nodes = np.array(random.sample(set(range(data.num_nodes)) - set(train_nodes), 500))
        test_nodes = np.array(random.sample(set(range(data.num_nodes)) - set(train_nodes) - set(val_nodes), 1000))

    if dataset_name == 'wiki':
        print('is reading {} dataset...'.format(dataset_name))
        df = pd.read_csv('../data/wiki/graph.txt', header=None, sep='\t', names=['source', 'target'])
        graph = nx.from_pandas_edgelist(df=df, source='source', target='target', edge_attr=None)

        df_label = pd.read_csv('../data/wiki/group.txt', header=None, sep='\t', names=['node_id', 'label'])
        available_nodes = df_label['node_id'].unique()

        graph = graph.subgraph(available_nodes)
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))

        graph = nx.relabel_nodes(graph, mapping, copy=True)
        df_label['node_id'] = df_label['node_id'].replace(mapping)
        df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

        # ecode label into numeric and set them in order
        le = preprocessing.LabelEncoder()
        df_label['label'] = le.fit_transform(df_label['label'])

        if use_features:
            feature = np.zeros((df_label['node_id'].nunique(), 4973))
            with open('../data/wiki/tfidf.txt') as f:
                for line in f:
                    id_1, id_2, value = line.split('\t')
                    feature[int(id_1)][int(id_2)] = value
            feature = torch.FloatTensor(feature)
        else:
            feature = torch.FloatTensor(np.identity(graph.number_of_nodes()))

        data = tg.data.Data(x=feature,
                            y=torch.Tensor(df_label['label']).long(),
                            edge_index=tg.utils.from_networkx(G=graph).edge_index)

        # some class of wiki dataset doesn't have 10/20 samples
        non_train_nodes = random.sample(list(range(data.num_nodes)), int(data.num_nodes * ((1 - 20 / 100))))
        train_nodes = np.array(list(set(range(data.num_nodes)) - set(non_train_nodes)))
        val_nodes = np.array(non_train_nodes[:int(data.num_nodes * ((1 - 20 / 100)) / 2)])
        test_nodes = np.array(non_train_nodes[int(data.num_nodes * ((1 - 20 / 100)) / 2):])
        # train_nodes = np.empty(0, dtype=int)
        # for label in data.y.unique():
        #     index = torch.nonzero(data.y==label).view(-1)
        #     train_nodes = np.append(train_nodes, np.array(random.sample(index.data.numpy().tolist(), 5)))
        # val_nodes = np.array(random.sample(set(range(data.num_nodes))-set(train_nodes), 500))
        # test_nodes = np.array(random.sample(set(range(data.num_nodes))-set(train_nodes)-set(val_nodes), 1000))

    elif dataset_name == 'acm':
        print('is reading {} dataset...'.format(dataset_name))
        path = '../data/acm/acm_graph.txt'
        data = np.loadtxt('../data/acm/acm.txt')
        N = data.shape[0]
        idx = np.array([i for i in range(N)], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(path, dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(N, N), dtype=np.float32)
        graph = nx.from_scipy_sparse_matrix(adj)

        df_label = pd.read_csv('../data/acm/acm_label.txt', header=None).reset_index()
        df_label.columns = ['node_id', 'label']
        available_nodes = df_label['node_id'].unique()

        graph = graph.subgraph(available_nodes)
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))

        graph = nx.relabel_nodes(graph, mapping, copy=True)
        df_label['node_id'] = df_label['node_id'].replace(mapping)
        df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

        # ecode label into numeric and set them in order
        le = preprocessing.LabelEncoder()
        df_label['label'] = le.fit_transform(df_label['label'])

        if use_features:
            feature = np.loadtxt('../data/acm/acm.txt')
            feature = torch.FloatTensor(feature)
        else:
            feature = torch.FloatTensor(np.identity(graph.number_of_nodes()))

        data = tg.data.Data(x=feature,
                            y=torch.Tensor(df_label['label']).long(),
                            edge_index=tg.utils.from_networkx(G=graph).edge_index)

        train_nodes = np.empty(0, dtype=int)
        for label in data.y.unique():
            index = torch.nonzero(data.y == label).view(-1)
            train_nodes = np.append(train_nodes, np.array(random.sample(index.data.numpy().tolist(), 20)))
        val_nodes = np.array(random.sample(set(range(data.num_nodes)) - set(train_nodes), 500))
        test_nodes = np.array(random.sample(set(range(data.num_nodes)) - set(train_nodes) - set(val_nodes), 1000))

    elif dataset_name == 'dblp':
        print('is reading {} dataset...'.format(dataset_name))
        path = '../data/dblp/dblp_graph.txt'
        data = np.loadtxt('../data/dblp/dblp.txt')
        N = data.shape[0]
        idx = np.array([i for i in range(N)], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(path, dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(N, N), dtype=np.float32).toarray()
        adj[-1][-1] = 1
        # graph = nx.from_scipy_sparse_matrix(adj)
        graph = nx.from_numpy_array(adj)

        df_label = pd.read_csv('../data/dblp/dblp_label.txt', header=None).reset_index()
        df_label.columns = ['node_id', 'label']
        available_nodes = df_label['node_id'].unique()

        graph = graph.subgraph(available_nodes)
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))

        graph = nx.relabel_nodes(graph, mapping, copy=True)
        df_label['node_id'] = df_label['node_id'].replace(mapping)
        df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

        # ecode label into numeric and set them in order
        le = preprocessing.LabelEncoder()
        df_label['label'] = le.fit_transform(df_label['label'])

        if use_features:
            feature = np.loadtxt('../data/dblp/dblp.txt')
            feature = torch.FloatTensor(feature)
        else:
            feature = torch.FloatTensor(np.identity(graph.number_of_nodes()))

        data = tg.data.Data(x=feature,
                            y=torch.Tensor(df_label['label']).long(),
                            edge_index=tg.utils.from_networkx(G=graph).edge_index)

        train_nodes = np.empty(0, dtype=int)
        for label in data.y.unique():
            index = torch.nonzero(data.y == label).view(-1)
            train_nodes = np.append(train_nodes, np.array(random.sample(index.data.numpy().tolist(), 20)))
        val_nodes = np.array(random.sample(set(range(data.num_nodes)) - set(train_nodes), 500))
        test_nodes = np.array(random.sample(set(range(data.num_nodes)) - set(train_nodes) - set(val_nodes), 1000))

    elif dataset_name == 'karate':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.KarateClub(transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()
        non_train_nodes = random.sample(list(range(data.num_nodes)), int(data.num_nodes * ((1 - 20 / 100))))
        train_nodes = np.array(list(set(range(data.num_nodes)) - set(non_train_nodes)))
        val_nodes = np.array(non_train_nodes[:int(data.num_nodes * ((1 - 20 / 100)) / 2)])
        test_nodes = np.array(non_train_nodes[int(data.num_nodes * ((1 - 20 / 100)) / 2):])

    if dataset_name not in ['cora', 'citeseer', 'pubmed']:
        train_mask = np.array([False] * data.num_nodes)
        val_mask = np.array([False] * data.num_nodes)
        test_mask = np.array([False] * data.num_nodes)
        train_mask[train_nodes] = True
        val_mask[val_nodes] = True
        test_mask[test_nodes] = True

        data.train_mask = torch.BoolTensor(train_mask)
        data.val_mask = torch.BoolTensor(val_mask)
        data.test_mask = torch.BoolTensor(test_mask)
    # if tg.utils.contains_isolated_nodes(data.edge_index):
    #     data.edge_index = tg.utils.remove_isolated_nodes(data.edge_index, num_nodes=data.num_nodes)[0]
    #     data.x = data.x[data.edge_index.unique()]

    return data


def get_LP_dataset(dataset_name, use_features, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if dataset_name == 'emails':
        print('is reading {} dataset...'.format(dataset_name))
        df = pd.read_csv('../data/emails/email.txt', header=None, sep=' ', names=['source', 'target'])
        graph = nx.from_pandas_edgelist(df=df, source='source', target='target', edge_attr=None)

        df_label = pd.read_csv('../data/emails/email_labels.txt', header=None, sep=' ', names=['node_id', 'label'])
        df_label = df_label[
            df_label['label'].isin(df_label['label'].value_counts()[df_label['label'].value_counts() > 20].index)]
        available_nodes = df_label['node_id'].unique()

        graph = graph.subgraph(available_nodes)
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))

        graph = nx.relabel_nodes(graph, mapping, copy=True)
        df_label['node_id'] = df_label['node_id'].replace(mapping)
        df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

        # ecode label into numeric and set them in order
        le = preprocessing.LabelEncoder()
        df_label['label'] = le.fit_transform(df_label['label'])

        identify_oh_feature = torch.FloatTensor(np.identity(graph.number_of_nodes()))

        data = tg.data.Data(
            x=identify_oh_feature,
            y=torch.Tensor(df_label['label']).long(),
            edge_index=tg.utils.from_networkx(G=graph).edge_index
        )

    elif dataset_name == 'cora':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Planetoid(root='../input/', name='Cora', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()

    elif dataset_name == 'citeseer':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Planetoid(
            root='../input/', name='CiteSeer', transform=T.NormalizeFeatures(), pre_transform=None
        )[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()

    elif dataset_name == 'pubmed':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Planetoid(
            root='../input/', name='PubMed', transform=T.NormalizeFeatures(), pre_transform=None
        )[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()

    elif dataset_name == 'cs':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Coauthor(root='../input/', name='CS', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()

    elif dataset_name == 'physic':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Coauthor(root='../input/', name='Physics', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()

    elif dataset_name == 'computers':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Amazon(root='../input/', name='Computers', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()

    elif dataset_name == 'photo':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Amazon(root='../input/', name='Photo', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()

    if dataset_name == 'wiki':
        print('is reading {} dataset...'.format(dataset_name))
        df = pd.read_csv('../data/wiki/graph.txt', header=None, sep='\t', names=['source', 'target'])
        graph = nx.from_pandas_edgelist(df=df, source='source', target='target', edge_attr=None)

        df_label = pd.read_csv('../data/wiki/group.txt', header=None, sep='\t', names=['node_id', 'label'])
        available_nodes = df_label['node_id'].unique()

        graph = graph.subgraph(available_nodes)
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))

        graph = nx.relabel_nodes(graph, mapping, copy=True)
        df_label['node_id'] = df_label['node_id'].replace(mapping)
        df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

        # ecode label into numeric and set them in order
        le = preprocessing.LabelEncoder()
        df_label['label'] = le.fit_transform(df_label['label'])

        if use_features:
            feature = np.zeros((df_label['node_id'].nunique(), 4973))
            with open('../data/wiki/tfidf.txt') as f:
                for line in f:
                    id_1, id_2, value = line.split('\t')
                    feature[int(id_1)][int(id_2)] = value
            feature = torch.FloatTensor(feature)
        else:
            feature = torch.FloatTensor(np.identity(graph.number_of_nodes()))

        data = tg.data.Data(
            x=feature,
            y=torch.Tensor(df_label['label']).long(),
            edge_index=tg.utils.from_networkx(G=graph).edge_index
        )

    elif dataset_name == 'acm':
        print('is reading {} dataset...'.format(dataset_name))
        path = '../data/acm/acm_graph.txt'
        data = np.loadtxt('../data/acm/acm.txt')
        N = data.shape[0]
        idx = np.array([i for i in range(N)], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(path, dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(N, N), dtype=np.float32)
        graph = nx.from_scipy_sparse_matrix(adj)

        df_label = pd.read_csv('../data/acm/acm_label.txt', header=None).reset_index()
        df_label.columns = ['node_id', 'label']
        available_nodes = df_label['node_id'].unique()

        graph = graph.subgraph(available_nodes)
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))

        graph = nx.relabel_nodes(graph, mapping, copy=True)
        df_label['node_id'] = df_label['node_id'].replace(mapping)
        df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

        # ecode label into numeric and set them in order
        le = preprocessing.LabelEncoder()
        df_label['label'] = le.fit_transform(df_label['label'])

        if use_features:
            feature = np.loadtxt('../data/acm/acm.txt')
            feature = torch.FloatTensor(feature)
        else:
            feature = torch.FloatTensor(np.identity(graph.number_of_nodes()))

        data = tg.data.Data(
            x=feature,
            y=torch.Tensor(df_label['label']).long(),
            edge_index=tg.utils.from_networkx(G=graph).edge_index
        )

    elif dataset_name == 'dblp':
        print('is reading {} dataset...'.format(dataset_name))
        path = '../data/dblp/dblp_graph.txt'
        data = np.loadtxt('../data/dblp/dblp.txt')
        N = data.shape[0]
        idx = np.array([i for i in range(N)], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt(path, dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(N, N), dtype=np.float32).toarray()
        adj[-1][-1] = 1
        # graph = nx.from_scipy_sparse_matrix(adj)
        graph = nx.from_numpy_array(adj)

        df_label = pd.read_csv('../data/dblp/dblp_label.txt', header=None).reset_index()
        df_label.columns = ['node_id', 'label']
        available_nodes = df_label['node_id'].unique()

        graph = graph.subgraph(available_nodes)
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))

        graph = nx.relabel_nodes(graph, mapping, copy=True)
        df_label['node_id'] = df_label['node_id'].replace(mapping)
        df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

        # ecode label into numeric and set them in order
        le = preprocessing.LabelEncoder()
        df_label['label'] = le.fit_transform(df_label['label'])

        if use_features:
            feature = np.loadtxt('../data/dblp/dblp.txt')
            feature = torch.FloatTensor(feature)
        else:
            feature = torch.FloatTensor(np.identity(graph.number_of_nodes()))

        data = tg.data.Data(
            x=feature,
            y=torch.Tensor(df_label['label']).long(),
            edge_index=tg.utils.from_networkx(G=graph).edge_index
        )

    elif dataset_name == 'karate':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.KarateClub(transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()

    # release gpu memory
    torch.cuda.empty_cache()
    return data
