import numpy as np
import scipy.sparse as sp
import pandas as pd
import networkx as nx
import random
from sklearn import preprocessing

import torch_geometric as tg
import torch_geometric.transforms as T
import torch

def get_GC_dataset(dataset_name, use_features, batch_size, local_test, SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
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

# def get_GC_dataset(dataset_name, use_features, SEED):
#     random.seed(SEED)
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)

#     dataset = tg.datasets.TUDataset('../input/', name=dataset_name, transform=T.NormalizeFeatures(), pre_filter=None)
#     dataset = dataset.shuffle()
#     n = (len(dataset) + 9) // 10
#     test_dataset = dataset[:n]
#     val_dataset = dataset[n:2 * n]
#     train_dataset = dataset[2 * n:]
#     # test_loader = tg.data.DenseDataLoader(test_dataset, batch_size=20)
#     # val_loader = tg.data.DenseDataLoader(val_dataset, batch_size=20)
#     # train_loader = tg.data.DenseDataLoader(train_dataset, batch_size=20)

#     return train_dataset, val_dataset, test_dataset

def get_NC_dataset(dataset_name, use_features, SEED, num_class=None):
    random.seed(SEED)
    np.random.seed(SEED)

    if dataset_name == 'emails':
        print('is reading {} dataset...'.format(dataset_name))
        df = pd.read_csv('../data/emails/email.txt', header=None, sep=' ', names=['source', 'target'])
        graph = nx.from_pandas_edgelist(df=df, source='source', target='target', edge_attr=None)

        df_label = pd.read_csv('../data/emails/email_labels.txt', header=None, sep=' ', names=['node_id', 'label'])
        df_label = df_label[df_label['label'].isin(df_label['label'].value_counts()[df_label['label'].value_counts()>20].index)]
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

        data = tg.data.Data(x=identify_oh_feature,
                        y=torch.Tensor(df_label['label']).long(),
                        edge_index=tg.utils.from_networkx(G=graph).edge_index)
        
        non_train_nodes = random.sample(list(range(data.num_nodes)), int(data.num_nodes*((1-20/100))))
        train_nodes = np.array(list(set(range(data.num_nodes))-set(non_train_nodes)))
        val_nodes = np.array(non_train_nodes[:int(data.num_nodes*((1-20/100))/2)])
        test_nodes = np.array(non_train_nodes[int(data.num_nodes*((1-20/100))/2):])

        train_mask = np.array([False] * data.num_nodes)
        val_mask = np.array([False] * data.num_nodes)
        test_mask = np.array([False] * data.num_nodes)
        train_mask[train_nodes] = True
        val_mask[val_nodes] = True
        test_mask[test_nodes] = True

        data.train_mask = torch.Tensor(train_mask).bool()
        data.val_mask = torch.Tensor(val_mask).bool()
        data.test_mask = torch.Tensor(test_mask).bool()
        
        ls_data = [data]

    elif dataset_name == 'dphi_1':
        print('is reading {} dataset...'.format(dataset_name))
        df_association = pd.read_csv('../data/DPHI/bio-pathways-associations.csv')
        df_class = pd.read_csv('../data/DPHI/bio-pathways-diseaseclasses.csv')

        selected_class = df_class['Disease Class'].value_counts().index[:num_class].values
        
        selected_disease = df_class[df_class['Disease Class'].isin(selected_class)]['Disease Name'].values
        df_association = df_association[df_association['Disease Name'].isin(selected_disease)].reset_index(drop=True)

        selected_gene = []
        for items in df_association['Associated Gene IDs'].values:
            selected_gene += items.split(', ')
        selected_gene = list(set(selected_gene))
        selected_gene = [int(item) for item in selected_gene]

        df_network = pd.read_csv('../data/DPHI/bio-pathways-network.csv')
        graph = nx.from_pandas_edgelist(df=df_network, source='Gene ID 1', target='Gene ID 2')
        graph = graph.subgraph(selected_gene)
        giant = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(giant)
        selected_gene = list(graph.nodes)

        df_gene_class = pd.merge(df_association, df_class, on='Disease Name')[['Disease Class', 'Associated Gene IDs']]
        df_gene_class['Associated Gene IDs'] = df_gene_class['Associated Gene IDs'].apply(lambda item: item.split(', '))
        df_gene_class = df_gene_class.explode('Associated Gene IDs').drop_duplicates().reset_index(drop=True)
        df_gene_class['Associated Gene IDs'] = df_gene_class['Associated Gene IDs'].apply(int)
        df_gene_class = df_gene_class[df_gene_class['Associated Gene IDs'].isin(selected_gene)].reset_index(drop=True)

        keys = df_gene_class['Disease Class'].unique().tolist()
        vals = range(df_gene_class['Disease Class'].nunique())
        mapping = dict(zip(keys, vals))
        print(mapping)
        df_gene_class['Disease Class'] = df_gene_class['Disease Class'].replace(mapping)

        df_label = df_gene_class.groupby('Associated Gene IDs')['Disease Class'].apply(list).reset_index()
        df_label = df_label.rename(columns={'Associated Gene IDs': "node_id", 'Disease Class': 'label'})
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        df_label['label'] = mlb.fit_transform(df_label['label'].values.tolist()).tolist()

        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))

        graph = nx.relabel_nodes(graph, mapping, copy=True)
        df_label['node_id'] = df_label['node_id'].replace(mapping)
        df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

        identify_oh_feature = torch.FloatTensor(np.identity(graph.number_of_nodes()))

        data = tg.data.Data(x=identify_oh_feature,
                        y=torch.Tensor(df_label['label']).float(),
                        edge_index=tg.utils.from_networkx(G=graph).edge_index)
        
        non_train_nodes = random.sample(list(range(data.num_nodes)), int(data.num_nodes*((1-20/100))))
        train_nodes = np.array(list(set(range(data.num_nodes))-set(non_train_nodes)))
        val_nodes = np.array(non_train_nodes[:int(data.num_nodes*((1-20/100))/2)])
        test_nodes = np.array(non_train_nodes[int(data.num_nodes*((1-20/100))/2):])

        train_mask = np.array([False] * data.num_nodes)
        val_mask = np.array([False] * data.num_nodes)
        test_mask = np.array([False] * data.num_nodes)
        train_mask[train_nodes] = True
        val_mask[val_nodes] = True
        test_mask[test_nodes] = True

        data.train_mask = torch.Tensor(train_mask).bool()
        data.val_mask = torch.Tensor(val_mask).bool()
        data.test_mask = torch.Tensor(test_mask).bool()
        
        ls_data = [data]

    elif dataset_name == 'dphi_2':
        print('is reading {} dataset...'.format(dataset_name))
        df_association = pd.read_csv('../data/DPHI/bio-pathways-associations.csv')
        df_class = pd.read_csv('../data/DPHI/bio-pathways-diseaseclasses.csv')

        selected_class = df_class['Disease Class'].value_counts().index[:num_class].values

        selected_disease = df_class[df_class['Disease Class'].isin(selected_class)]['Disease Name'].values
        df_association = df_association[df_association['Disease Name'].isin(selected_disease)].reset_index(drop=True)

        selected_gene = []
        for items in df_association['Associated Gene IDs'].values:
            selected_gene += items.split(', ')
        selected_gene = list(set(selected_gene))
        selected_gene = [int(item) for item in selected_gene]

        df_network = pd.read_csv('../data/DPHI/bio-pathways-network.csv')
        graph = nx.from_pandas_edgelist(df=df_network, source='Gene ID 1', target='Gene ID 2')
        graph = graph.subgraph(selected_gene)
        giant = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(giant)
        selected_gene = list(graph.nodes)

        df_gene_class = pd.merge(df_association, df_class, on='Disease Name')[['Disease Class', 'Associated Gene IDs']]
        df_gene_class['Associated Gene IDs'] = df_gene_class['Associated Gene IDs'].apply(lambda item: item.split(', '))
        df_gene_class = df_gene_class.explode('Associated Gene IDs').drop_duplicates().reset_index(drop=True)
        df_gene_class['Associated Gene IDs'] = df_gene_class['Associated Gene IDs'].apply(int)
        df_gene_class = df_gene_class[df_gene_class['Associated Gene IDs'].isin(selected_gene)].reset_index(drop=True)

        keys = df_gene_class['Disease Class'].unique().tolist()
        vals = range(df_gene_class['Disease Class'].nunique())
        mapping = dict(zip(keys, vals))
        print(mapping)
        df_gene_class['Disease Class'] = df_gene_class['Disease Class'].replace(mapping)

        df_label = df_gene_class.groupby('Associated Gene IDs')['Disease Class'].apply(list).reset_index()
        df_label = df_label.rename(columns={'Associated Gene IDs': "node_id", 'Disease Class': 'label'})

        temp = df_association[['Associated Gene IDs', 'Disease Name']]
        temp['Associated Gene IDs'] = temp['Associated Gene IDs'].apply(lambda item: item.split(', '))
        temp = temp.explode('Associated Gene IDs').drop_duplicates().reset_index(drop=True)
        temp['Associated Gene IDs'] = temp['Associated Gene IDs'].apply(int)


        selected_disease = temp['Disease Name'].value_counts()[temp['Disease Name'].value_counts()>200].index.values
        temp = temp[temp['Disease Name'].isin(selected_disease)]

        keys = temp['Disease Name'].unique().tolist()
        vals = range(temp['Disease Name'].nunique())
        mapping = dict(zip(keys, vals))
        print(len(mapping), mapping)
        temp['Disease Name'] = temp['Disease Name'].replace(mapping)

        temp = temp[temp['Associated Gene IDs'].isin(df_label['node_id'].unique())]

        temp = pd.merge(df_label, temp, left_on='node_id', right_on='Associated Gene IDs', how='left')[['node_id', 'Disease Name']]
        available_nodes = temp[~temp['Disease Name'].isna()]['node_id'].unique()

        graph = graph.subgraph(available_nodes)
        giant = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(giant)
        selected_gene = list(graph.nodes)

        temp = temp[temp['node_id'].isin(selected_gene)]

        # df_label = temp.groupby('Associated Gene IDs')['Disease Name'].apply(list).reset_index()
        # df_label = df_label.rename(columns={'Associated Gene IDs': "node_id", 'Disease Name': 'label'})
        df_label = temp.groupby('node_id')['Disease Name'].apply(list).reset_index()
        df_label = df_label.rename(columns={'Disease Name': 'label'})

        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        df_label['label'] = mlb.fit_transform(df_label['label'].values.tolist()).tolist()


        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))

        graph = nx.relabel_nodes(graph, mapping, copy=True)
        df_label['node_id'] = df_label['node_id'].replace(mapping)
        df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

        identify_oh_feature = torch.FloatTensor(np.identity(graph.number_of_nodes()))

        data = tg.data.Data(x=identify_oh_feature,
                        y=torch.Tensor(df_label['label']).float(),
                        edge_index=tg.utils.from_networkx(G=graph).edge_index)

        non_train_nodes = random.sample(list(range(data.num_nodes)), int(data.num_nodes*((1-20/100))))
        train_nodes = np.array(list(set(range(data.num_nodes))-set(non_train_nodes)))
        val_nodes = np.array(non_train_nodes[:int(data.num_nodes*((1-20/100))/2)])
        test_nodes = np.array(non_train_nodes[int(data.num_nodes*((1-20/100))/2):])

        train_mask = np.array([False] * data.num_nodes)
        val_mask = np.array([False] * data.num_nodes)
        test_mask = np.array([False] * data.num_nodes)
        train_mask[train_nodes] = True
        val_mask[val_nodes] = True
        test_mask[test_nodes] = True

        data.train_mask = torch.Tensor(train_mask).bool()
        data.val_mask = torch.Tensor(val_mask).bool()
        data.test_mask = torch.Tensor(test_mask).bool()

        ls_data = [data]
        # df_association = pd.read_csv('../data/DPHI/bio-pathways-associations.csv')
        # df_class = pd.read_csv('../data/DPHI/bio-pathways-diseaseclasses.csv')

        # selected_class = df_class['Disease Class'].value_counts().index[:num_class].values

        # selected_disease = df_class[df_class['Disease Class'].isin(selected_class)]['Disease Name'].values
        # df_association = df_association[df_association['Disease Name'].isin(selected_disease)].reset_index(drop=True)

        # selected_gene = []
        # for items in df_association['Associated Gene IDs'].values:
        #     selected_gene += items.split(', ')
        # selected_gene = list(set(selected_gene))
        # selected_gene = [int(item) for item in selected_gene]

        # df_network = pd.read_csv('../data/DPHI/bio-pathways-network.csv')
        # graph = nx.from_pandas_edgelist(df=df_network, source='Gene ID 1', target='Gene ID 2')
        # graph = graph.subgraph(selected_gene)
        # giant = max(nx.connected_components(graph), key=len)
        # graph = graph.subgraph(giant)
        # selected_gene = list(graph.nodes)

        # df_gene_class = pd.merge(df_association, df_class, on='Disease Name')[['Disease Class', 'Associated Gene IDs']]
        # df_gene_class['Associated Gene IDs'] = df_gene_class['Associated Gene IDs'].apply(lambda item: item.split(', '))
        # df_gene_class = df_gene_class.explode('Associated Gene IDs').drop_duplicates().reset_index(drop=True)
        # df_gene_class['Associated Gene IDs'] = df_gene_class['Associated Gene IDs'].apply(int)
        # df_gene_class = df_gene_class[df_gene_class['Associated Gene IDs'].isin(selected_gene)].reset_index(drop=True)

        # keys = df_gene_class['Disease Class'].unique().tolist()
        # vals = range(df_gene_class['Disease Class'].nunique())
        # mapping = dict(zip(keys, vals))
        # print(mapping)
        # df_gene_class['Disease Class'] = df_gene_class['Disease Class'].replace(mapping)

        # df_label = df_gene_class.groupby('Associated Gene IDs')['Disease Class'].apply(list).reset_index()
        # df_label = df_label.rename(columns={'Associated Gene IDs': "node_id", 'Disease Class': 'label'})

        # temp = df_association[['Associated Gene IDs', 'Disease Name']]
        # temp['Associated Gene IDs'] = temp['Associated Gene IDs'].apply(lambda item: item.split(', '))
        # temp = temp.explode('Associated Gene IDs').drop_duplicates().reset_index(drop=True)
        # temp['Associated Gene IDs'] = temp['Associated Gene IDs'].apply(int)

        # keys = temp['Disease Name'].unique().tolist()
        # vals = range(temp['Disease Name'].nunique())
        # mapping = dict(zip(keys, vals))
        # print(mapping)
        # temp['Disease Name'] = temp['Disease Name'].replace(mapping)

        # temp = temp[temp['Associated Gene IDs'].isin(df_label['node_id'].unique())]

        # df_label = temp.groupby('Associated Gene IDs')['Disease Name'].apply(list).reset_index()
        # df_label = df_label.rename(columns={'Associated Gene IDs': "node_id", 'Disease Name': 'label'})
        
        # from sklearn.preprocessing import MultiLabelBinarizer
        # mlb = MultiLabelBinarizer()
        # df_label['label'] = mlb.fit_transform(df_label['label'].values.tolist()).tolist()

        # keys = list(graph.nodes)
        # vals = range(graph.number_of_nodes())
        # mapping = dict(zip(keys, vals))

        # graph = nx.relabel_nodes(graph, mapping, copy=True)
        # df_label['node_id'] = df_label['node_id'].replace(mapping)
        # df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

        # identify_oh_feature = torch.FloatTensor(np.identity(graph.number_of_nodes()))

        # data = tg.data.Data(x=identify_oh_feature,
        #                 y=torch.Tensor(df_label['label']).float(),
        #                 edge_index=tg.utils.from_networkx(G=graph).edge_index)
        
        # non_train_nodes = random.sample(list(range(data.num_nodes)), int(data.num_nodes*((1-20/100))))
        # train_nodes = np.array(list(set(range(data.num_nodes))-set(non_train_nodes)))
        # val_nodes = np.array(non_train_nodes[:int(data.num_nodes*((1-20/100))/2)])
        # test_nodes = np.array(non_train_nodes[int(data.num_nodes*((1-20/100))/2):])

        # train_mask = np.array([False] * data.num_nodes)
        # val_mask = np.array([False] * data.num_nodes)
        # test_mask = np.array([False] * data.num_nodes)
        # train_mask[train_nodes] = True
        # val_mask[val_nodes] = True
        # test_mask[test_nodes] = True

        # data.train_mask = torch.Tensor(train_mask).bool()
        # data.val_mask = torch.Tensor(val_mask).bool()
        # data.test_mask = torch.Tensor(test_mask).bool()
        
        # ls_data = [data]

    # elif dataset_name == 'dphi_2':
    #     print('is reading {} dataset...'.format(dataset_name))
    #     df_association = pd.read_csv('../data/DPHI/bio-pathways-associations.csv')
    #     df_class = pd.read_csv('../data/DPHI/bio-pathways-diseaseclasses.csv')

    #     selected_class = df_class['Disease Class'].value_counts().index[:5].values

    #     selected_disease = df_class[df_class['Disease Class'].isin(selected_class)]['Disease Name'].values
    #     df_association = df_association[df_association['Disease Name'].isin(selected_disease)].reset_index(drop=True)

    #     selected_gene = []
    #     for items in df_association['Associated Gene IDs'].values:
    #         selected_gene += items.split(', ')
    #     selected_gene = list(set(selected_gene))
    #     selected_gene = [int(item) for item in selected_gene]

    #     df_network = pd.read_csv('../data/DPHI/bio-pathways-network.csv')
    #     graph = nx.from_pandas_edgelist(df=df_network, source='Gene ID 1', target='Gene ID 2')
    #     graph = graph.subgraph(selected_gene)
    #     giant = max(nx.connected_components(graph), key=len)
    #     graph = graph.subgraph(giant)
    #     selected_gene = list(graph.nodes)

    #     df_gene_class = pd.merge(df_association, df_class, on='Disease Name')[['Disease Class', 'Associated Gene IDs']]
    #     df_gene_class['Associated Gene IDs'] = df_gene_class['Associated Gene IDs'].apply(lambda item: item.split(', '))
    #     df_gene_class = df_gene_class.explode('Associated Gene IDs').drop_duplicates().reset_index(drop=True)
    #     df_gene_class['Associated Gene IDs'] = df_gene_class['Associated Gene IDs'].apply(int)
    #     df_gene_class = df_gene_class[df_gene_class['Associated Gene IDs'].isin(selected_gene)].reset_index(drop=True)

    #     keys = df_gene_class['Disease Class'].unique().tolist()
    #     vals = range(df_gene_class['Disease Class'].nunique())
    #     mapping = dict(zip(keys, vals))
    #     print(mapping)
    #     df_gene_class['Disease Class'] = df_gene_class['Disease Class'].replace(mapping)

    #     df_label = df_gene_class.groupby('Associated Gene IDs')['Disease Class'].apply(list).reset_index()
    #     df_label = df_label.rename(columns={'Associated Gene IDs': "node_id", 'Disease Class': 'label'})

    #     temp = df_association[['Associated Gene IDs', 'Disease Name']]
    #     temp['Associated Gene IDs'] = temp['Associated Gene IDs'].apply(lambda item: item.split(', '))
    #     temp = temp.explode('Associated Gene IDs').drop_duplicates().reset_index(drop=True)
    #     temp['Associated Gene IDs'] = temp['Associated Gene IDs'].apply(int)

    #     tt = temp['Disease Name'].value_counts()
    #     temp = temp[temp['Disease Name'].isin(tt[tt>100].index.values)]

    #     temp_count = temp['Associated Gene IDs'].value_counts()

    #     available_nodes = temp_count[temp_count==1].index.values
    #     available_nodes = list(set(df_label['node_id'].values) & set(available_nodes))

    #     df_gene_dis_name = temp[temp['Associated Gene IDs'].isin(available_nodes)].drop_duplicates()

    #     keys = df_gene_dis_name['Disease Name'].unique().tolist()
    #     vals = range(df_gene_dis_name['Disease Name'].nunique())
    #     mapping = dict(zip(keys, vals))
    #     df_gene_dis_name['Disease Name'] = df_gene_dis_name['Disease Name'].replace(mapping)

    #     df_label = pd.merge(df_label, df_gene_dis_name, left_on='node_id', right_on='Associated Gene IDs', how='left')[['node_id', 'Disease Name']]
    #     df_label = df_label.rename(columns={'Disease Name': 'label'}).reset_index(drop=True)
    #     df_label['label'] = df_label['label'].fillna(99)

    #     keys = list(graph.nodes)
    #     vals = range(graph.number_of_nodes())
    #     mapping = dict(zip(keys, vals))

    #     graph = nx.relabel_nodes(graph, mapping, copy=True)
    #     df_label['node_id'] = df_label['node_id'].replace(mapping)
    #     df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

    #     available_nodes = [mapping[item] for item in available_nodes]

    #     identify_oh_feature = torch.FloatTensor(np.identity(graph.number_of_nodes()))

    #     data = tg.data.Data(x=identify_oh_feature,
    #                         y=torch.Tensor(df_label['label']).long(),
    #                         edge_index=tg.utils.from_networkx(G=graph).edge_index,
    #                         available_nodes=available_nodes)
        
    #     non_train_nodes = random.sample(list(range(data.num_nodes)), int(data.num_nodes*((1-20/100))))
    #     train_nodes = np.array(list(set(range(data.num_nodes))-set(non_train_nodes)))
    #     val_nodes = np.array(non_train_nodes[:int(data.num_nodes*((1-20/100))/2)])
    #     test_nodes = np.array(non_train_nodes[int(data.num_nodes*((1-20/100))/2):])

    #     train_mask = np.array([False] * data.num_nodes)
    #     val_mask = np.array([False] * data.num_nodes)
    #     test_mask = np.array([False] * data.num_nodes)
    #     train_mask[train_nodes] = True
    #     val_mask[val_nodes] = True
    #     test_mask[test_nodes] = True

    #     data.train_mask = torch.Tensor(train_mask).bool()
    #     data.val_mask = torch.Tensor(val_mask).bool()
    #     data.test_mask = torch.Tensor(test_mask).bool()
        
    #     ls_data = [data]
    
    # elif dataset_name == 'disease_1':
    #     print('is reading {} dataset...'.format(dataset_name))
    #     df = pd.read_csv('../data/disease_nc/disease_nc.edges.csv', header=None, names=['source', 'target'])
    #     graph = nx.from_pandas_edgelist(df=df, source='source', target='target', edge_attr=None)

    #     array_label = np.load('../data/disease_nc/disease_nc.labels.npy')
    #     df_label = pd.DataFrame({'node_id': list(range(array_label.shape[0])), 'label': array_label})
    #     available_nodes = df_label['node_id'].unique()

    #     graph = graph.subgraph(available_nodes)
    #     keys = list(graph.nodes)
    #     vals = range(graph.number_of_nodes())
    #     mapping = dict(zip(keys, vals))

    #     graph = nx.relabel_nodes(graph, mapping, copy=True)
    #     df_label['node_id'] = df_label['node_id'].replace(mapping)
    #     df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

    #     # ecode label into numeric and set them in order
    #     le = preprocessing.LabelEncoder()
    #     df_label['label'] = le.fit_transform(df_label['label'])

    #     if not use_features:
    #         feature = torch.FloatTensor(np.identity(graph.number_of_nodes()))
    #     else:
    #         feat_npz = np.load('../data/disease_nc/disease_nc.feats.npz')
    #         feature = feat_npz['data'].reshape(feat_npz['shape'])
    #         feature = torch.FloatTensor(feature)

    #     data = tg.data.Data(x=feature,
    #                     y=torch.Tensor(df_label['label']).long(),
    #                     edge_index=tg.utils.from_networkx(G=graph).edge_index)
        
    #     non_train_nodes = random.sample(list(range(data.num_nodes)), int(data.num_nodes*((1-20/100))))
    #     train_nodes = np.array(list(set(range(data.num_nodes))-set(non_train_nodes)))
    #     val_nodes = np.array(non_train_nodes[:int(data.num_nodes*((1-20/100))/2)])
    #     test_nodes = np.array(non_train_nodes[int(data.num_nodes*((1-20/100))/2):])

    #     train_mask = np.array([False] * data.num_nodes)
    #     val_mask = np.array([False] * data.num_nodes)
    #     test_mask = np.array([False] * data.num_nodes)
    #     train_mask[train_nodes] = True
    #     val_mask[val_nodes] = True
    #     test_mask[test_nodes] = True

    #     data.train_mask = torch.Tensor(train_mask).bool()
    #     data.val_mask = torch.Tensor(val_mask).bool()
    #     data.test_mask = torch.Tensor(test_mask).bool()
        
    #     ls_data = [data]

    # elif dataset_name == 'disease_2':
    #     print('is reading {} dataset...'.format(dataset_name))
    #     df = pd.read_csv('../data/disease_lp/disease_lp.edges.csv', header=None, names=['source', 'target'])
    #     graph = nx.from_pandas_edgelist(df=df, source='source', target='target', edge_attr=None)

    #     array_label = np.load('../data/disease_lp/disease_lp.labels.npy')
    #     df_label = pd.DataFrame({'node_id': list(range(array_label.shape[0])), 'label': array_label})
    #     available_nodes = df_label['node_id'].unique()

    #     graph = graph.subgraph(available_nodes)
    #     keys = list(graph.nodes)
    #     vals = range(graph.number_of_nodes())
    #     mapping = dict(zip(keys, vals))

    #     graph = nx.relabel_nodes(graph, mapping, copy=True)
    #     df_label['node_id'] = df_label['node_id'].replace(mapping)
    #     df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

    #     # ecode label into numeric and set them in order
    #     le = preprocessing.LabelEncoder()
    #     df_label['label'] = le.fit_transform(df_label['label'])

    #     # if not use_features:
    #     feature = torch.FloatTensor(np.identity(graph.number_of_nodes()))
    #     # else:
    #     #     feat_npz = np.load('../data/disease_lp/disease_lp.feats.npz')
    #     #     feature = feat_npz['data'].reshape(feat_npz['shape'])
    #     #     feature = torch.FloatTensor(feature)

    #     data = tg.data.Data(x=feature,
    #                     y=torch.Tensor(df_label['label']).long(),
    #                     edge_index=tg.utils.from_networkx(G=graph).edge_index)
        
    #     non_train_nodes = random.sample(list(range(data.num_nodes)), int(data.num_nodes*((1-20/100))))
    #     train_nodes = np.array(list(set(range(data.num_nodes))-set(non_train_nodes)))
    #     val_nodes = np.array(non_train_nodes[:int(data.num_nodes*((1-20/100))/2)])
    #     test_nodes = np.array(non_train_nodes[int(data.num_nodes*((1-20/100))/2):])

    #     train_mask = np.array([False] * data.num_nodes)
    #     val_mask = np.array([False] * data.num_nodes)
    #     test_mask = np.array([False] * data.num_nodes)
    #     train_mask[train_nodes] = True
    #     val_mask[val_nodes] = True
    #     test_mask[test_nodes] = True

    #     data.train_mask = torch.Tensor(train_mask).bool()
    #     data.val_mask = torch.Tensor(val_mask).bool()
    #     data.test_mask = torch.Tensor(test_mask).bool()
        
    #     ls_data = [data]

    elif dataset_name == 'karate':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.KarateClub(transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()
        ls_data = [data]

    elif dataset_name == 'cora':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Planetoid(root='../input/', name='Cora', transform=T.NormalizeFeatures(), pre_transform=None)[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()
        ls_data = [data]
    
    elif dataset_name == 'citeseer':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Planetoid(root='../input/', name='CiteSeer', transform=T.NormalizeFeatures(), pre_transform=None)[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()
        ls_data = [data]

    elif dataset_name == 'pubmed':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Planetoid(root='../input/', name='PubMed', transform=T.NormalizeFeatures(), pre_transform=None)[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()
        ls_data = [data]
    
    elif dataset_name == 'cs':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Coauthor(root='../input/', name='CS', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()
        
        train_nodes = np.empty(0, dtype=int)
        for label in data.y.unique():
            index = torch.nonzero(data.y==label).view(-1)
            train_nodes = np.append(train_nodes, np.array(random.sample(index.data.numpy().tolist(), 20)))
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
        
        ls_data = [data]

    elif dataset_name == 'physic':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Coauthor(root='../input/', name='Physics', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()

        train_nodes = np.empty(0, dtype=int)
        for label in data.y.unique():
            index = torch.nonzero(data.y==label).view(-1)
            train_nodes = np.append(train_nodes, np.array(random.sample(index.data.numpy().tolist(), 20)))
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

        ls_data = [data]
    
    elif dataset_name == 'computers':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Amazon(root='../input/', name='Computers', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()

        train_nodes = np.empty(0, dtype=int)
        for label in data.y.unique():
            index = torch.nonzero(data.y==label).view(-1)
            train_nodes = np.append(train_nodes, np.array(random.sample(index.data.numpy().tolist(), 20)))
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

        ls_data = [data]

    elif dataset_name == 'photo':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Amazon(root='../input/', name='Photo', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()

        train_nodes = np.empty(0, dtype=int)
        for label in data.y.unique():
            index = torch.nonzero(data.y==label).view(-1)
            train_nodes = np.append(train_nodes, np.array(random.sample(index.data.numpy().tolist(), 20)))
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

        ls_data = [data]

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
        
        # some class of wiki dataset doesn't have 20 samples
        non_train_nodes = random.sample(list(range(data.num_nodes)), int(data.num_nodes*((1-20/100))))
        train_nodes = np.array(list(set(range(data.num_nodes))-set(non_train_nodes)))
        val_nodes = np.array(non_train_nodes[:int(data.num_nodes*((1-20/100))/2)])
        test_nodes = np.array(non_train_nodes[int(data.num_nodes*((1-20/100))/2):])

        train_mask = np.array([False] * data.num_nodes)
        val_mask = np.array([False] * data.num_nodes)
        test_mask = np.array([False] * data.num_nodes)
        train_mask[train_nodes] = True
        val_mask[val_nodes] = True
        test_mask[test_nodes] = True

        data.train_mask = torch.Tensor(train_mask).bool()
        data.val_mask = torch.Tensor(val_mask).bool()
        data.test_mask = torch.Tensor(test_mask).bool()
        
        ls_data = [data]
    
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
            index = torch.nonzero(data.y==label).view(-1)
            train_nodes = np.append(train_nodes, np.array(random.sample(index.data.numpy().tolist(), 20)))
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

        ls_data = [data]

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
            index = torch.nonzero(data.y==label).view(-1)
            train_nodes = np.append(train_nodes, np.array(random.sample(index.data.numpy().tolist(), 20)))
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

        ls_data = [data]

    elif dataset_name == 'karate':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.KarateClub(transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()

        non_train_nodes = random.sample(list(range(data.num_nodes)), int(data.num_nodes*((1-20/100))))
        train_nodes = np.array(list(set(range(data.num_nodes))-set(non_train_nodes)))
        val_nodes = np.array(non_train_nodes[:int(data.num_nodes*((1-20/100))/2)])
        test_nodes = np.array(non_train_nodes[int(data.num_nodes*((1-20/100))/2):])

        train_mask = np.array([False] * data.num_nodes)
        val_mask = np.array([False] * data.num_nodes)
        test_mask = np.array([False] * data.num_nodes)
        train_mask[train_nodes] = True
        val_mask[val_nodes] = True
        test_mask[test_nodes] = True

        data.train_mask = torch.Tensor(train_mask).bool()
        data.val_mask = torch.Tensor(val_mask).bool()
        data.test_mask = torch.Tensor(test_mask).bool()

        ls_data = [data]

    # release gpu memory
    torch.cuda.empty_cache()
    return ls_data



def get_LP_dataset(dataset_name, use_features, SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    if dataset_name == 'grid':
        print('is reading {} dataset...'.format(dataset_name))
        G = nx.grid_2d_graph(20, 20)
        G = nx.convert_node_labels_to_integers(G)
        keys = list(G.nodes)
        vals = range(G.number_of_nodes())
        mapping = dict(zip(keys, vals))
        G = nx.relabel_nodes(G, mapping, copy=True)
        identify_oh_feature = np.identity(G.number_of_nodes())
        graphs = [G]
        features = [identify_oh_feature]
        print('datatset reading is done.')
    
    elif dataset_name == 'emails':
        print('is reading {} dataset...'.format(dataset_name))
        df = pd.read_csv('../data/emails/email.txt', header=None, sep=' ', names=['source', 'target'])
        graph = nx.from_pandas_edgelist(df=df, source='source', target='target', edge_attr=None)

        df_label = pd.read_csv('../data/emails/email_labels.txt', header=None, sep=' ', names=['node_id', 'label'])
        df_label = df_label[df_label['label'].isin(df_label['label'].value_counts()[df_label['label'].value_counts()>20].index)]
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

        data = tg.data.Data(x=identify_oh_feature,
                        y=torch.Tensor(df_label['label']).long(),
                        edge_index=tg.utils.from_networkx(G=graph).edge_index)
        
        ls_data = [data]
    
    # elif dataset_name == 'dphi_1':
    #     print('is reading {} dataset...'.format(dataset_name))
    #     df_association = pd.read_csv('../data/DPHI/bio-pathways-associations.csv')
    #     df_class = pd.read_csv('../data/DPHI/bio-pathways-diseaseclasses.csv')

    #     selected_class = df_class['Disease Class'].value_counts().index[:num_class].values
        
    #     selected_disease = df_class[df_class['Disease Class'].isin(selected_class)]['Disease Name'].values
    #     df_association = df_association[df_association['Disease Name'].isin(selected_disease)].reset_index(drop=True)

    #     selected_gene = []
    #     for items in df_association['Associated Gene IDs'].values:
    #         selected_gene += items.split(', ')
    #     selected_gene = list(set(selected_gene))
    #     selected_gene = [int(item) for item in selected_gene]

    #     df_network = pd.read_csv('../data/DPHI/bio-pathways-network.csv')
    #     graph = nx.from_pandas_edgelist(df=df_network, source='Gene ID 1', target='Gene ID 2')
    #     graph = graph.subgraph(selected_gene)
    #     giant = max(nx.connected_components(graph), key=len)
    #     graph = graph.subgraph(giant)
    #     selected_gene = list(graph.nodes)

    #     df_gene_class = pd.merge(df_association, df_class, on='Disease Name')[['Disease Class', 'Associated Gene IDs']]
    #     df_gene_class['Associated Gene IDs'] = df_gene_class['Associated Gene IDs'].apply(lambda item: item.split(', '))
    #     df_gene_class = df_gene_class.explode('Associated Gene IDs').drop_duplicates().reset_index(drop=True)
    #     df_gene_class['Associated Gene IDs'] = df_gene_class['Associated Gene IDs'].apply(int)
    #     df_gene_class = df_gene_class[df_gene_class['Associated Gene IDs'].isin(selected_gene)].reset_index(drop=True)

    #     keys = df_gene_class['Disease Class'].unique().tolist()
    #     vals = range(df_gene_class['Disease Class'].nunique())
    #     mapping = dict(zip(keys, vals))
    #     print(mapping)
    #     df_gene_class['Disease Class'] = df_gene_class['Disease Class'].replace(mapping)

    #     df_label = df_gene_class.groupby('Associated Gene IDs')['Disease Class'].apply(list).reset_index()
    #     df_label = df_label.rename(columns={'Associated Gene IDs': "node_id", 'Disease Class': 'label'})
    #     from sklearn.preprocessing import MultiLabelBinarizer
    #     mlb = MultiLabelBinarizer()
    #     df_label['label'] = mlb.fit_transform(df_label['label'].values.tolist()).tolist()

    #     keys = list(graph.nodes)
    #     vals = range(graph.number_of_nodes())
    #     mapping = dict(zip(keys, vals))

    #     graph = nx.relabel_nodes(graph, mapping, copy=True)
    #     df_label['node_id'] = df_label['node_id'].replace(mapping)
    #     df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

    #     identify_oh_feature = torch.FloatTensor(np.identity(graph.number_of_nodes()))

    #     data = tg.data.Data(x=identify_oh_feature,
    #                     y=torch.Tensor(df_label['label']).float(),
    #                     edge_index=tg.utils.from_networkx(G=graph).edge_index)
        
    #     non_train_nodes = random.sample(list(range(data.num_nodes)), int(data.num_nodes*((1-20/100))))
    #     train_nodes = np.array(list(set(range(data.num_nodes))-set(non_train_nodes)))
    #     val_nodes = np.array(non_train_nodes[:int(data.num_nodes*((1-20/100))/2)])
    #     test_nodes = np.array(non_train_nodes[int(data.num_nodes*((1-20/100))/2):])

    #     train_mask = np.array([False] * data.num_nodes)
    #     val_mask = np.array([False] * data.num_nodes)
    #     test_mask = np.array([False] * data.num_nodes)
    #     train_mask[train_nodes] = True
    #     val_mask[val_nodes] = True
    #     test_mask[test_nodes] = True

    #     data.train_mask = torch.Tensor(train_mask).bool()
    #     data.val_mask = torch.Tensor(val_mask).bool()
    #     data.test_mask = torch.Tensor(test_mask).bool()
        
    #     ls_data = [data]

    # elif dataset_name == 'disease_1':
    #     print('is reading {} dataset...'.format(dataset_name))
    #     df = pd.read_csv('../data/disease_nc/disease_nc.edges.csv', header=None, names=['source', 'target'])
    #     graph = nx.from_pandas_edgelist(df=df, source='source', target='target', edge_attr=None)

    #     array_label = np.load('../data/disease_nc/disease_nc.labels.npy')
    #     df_label = pd.DataFrame({'node_id': list(range(array_label.shape[0])), 'label': array_label})
    #     available_nodes = df_label['node_id'].unique()

    #     graph = graph.subgraph(available_nodes)
    #     keys = list(graph.nodes)
    #     vals = range(graph.number_of_nodes())
    #     mapping = dict(zip(keys, vals))

    #     graph = nx.relabel_nodes(graph, mapping, copy=True)
    #     df_label['node_id'] = df_label['node_id'].replace(mapping)
    #     df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

    #     # ecode label into numeric and set them in order
    #     le = preprocessing.LabelEncoder()
    #     df_label['label'] = le.fit_transform(df_label['label'])

    #     if not use_features:
    #         feature = torch.FloatTensor(np.identity(graph.number_of_nodes()))
    #     else:
    #         feat_npz = np.load('../data/disease_nc/disease_nc.feats.npz')
    #         feature = feat_npz['data'].reshape(feat_npz['shape'])
    #         feature = torch.FloatTensor(feature)

    #     data = tg.data.Data(x=feature,
    #                     y=torch.Tensor(df_label['label']).long(),
    #                     edge_index=tg.utils.from_networkx(G=graph).edge_index)
        
    #     ls_data = [data]
    
    # elif dataset_name == 'disease_2':
    #     print('is reading {} dataset...'.format(dataset_name))
    #     df = pd.read_csv('../data/disease_lp/disease_lp.edges.csv', header=None, names=['source', 'target'])
    #     graph = nx.from_pandas_edgelist(df=df, source='source', target='target', edge_attr=None)

    #     array_label = np.load('../data/disease_lp/disease_lp.labels.npy')
    #     df_label = pd.DataFrame({'node_id': list(range(array_label.shape[0])), 'label': array_label})
    #     available_nodes = df_label['node_id'].unique()

    #     graph = graph.subgraph(available_nodes)
    #     keys = list(graph.nodes)
    #     vals = range(graph.number_of_nodes())
    #     mapping = dict(zip(keys, vals))

    #     graph = nx.relabel_nodes(graph, mapping, copy=True)
    #     df_label['node_id'] = df_label['node_id'].replace(mapping)
    #     df_label = df_label.sort_values('node_id', ascending=True).reset_index(drop=True)

    #     # ecode label into numeric and set them in order
    #     le = preprocessing.LabelEncoder()
    #     df_label['label'] = le.fit_transform(df_label['label'])

    #     # if not use_features:
    #     feature = torch.FloatTensor(np.identity(graph.number_of_nodes()))
    #     # else:
    #     #     feat_npz = np.load('../data/disease_nc/disease_nc.feats.npz')
    #     #     feature = feat_npz['data'].reshape(feat_npz['shape'])
    #     #     feature = torch.FloatTensor(feature)

    #     data = tg.data.Data(x=feature,
    #                     y=torch.Tensor(df_label['label']).long(),
    #                     edge_index=tg.utils.from_networkx(G=graph).edge_index)
        
    #     ls_data = [data]
    
    elif dataset_name == 'cora':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Planetoid(root='../input/', name='Cora', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()
        ls_data = [data]
    
    elif dataset_name == 'citeseer':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Planetoid(root='../input/', name='CiteSeer', transform=T.NormalizeFeatures(), pre_transform=None)[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()
        ls_data = [data]

    elif dataset_name == 'pubmed':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Planetoid(root='../input/', name='PubMed', transform=T.NormalizeFeatures(), pre_transform=None)[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()
        ls_data = [data]
    
    elif dataset_name == 'cs':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Coauthor(root='../input/', name='CS', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()
        
        ls_data = [data]

    elif dataset_name == 'physic':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Coauthor(root='../input/', name='Physics', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()

        ls_data = [data]
    
    elif dataset_name == 'computers':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Amazon(root='../input/', name='Computers', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()

        ls_data = [data]

    elif dataset_name == 'photo':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.Amazon(root='../input/', name='Photo', transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()

        ls_data = [data]

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
        
        ls_data = [data]
    
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

        ls_data = [data]

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

        ls_data = [data]

    elif dataset_name == 'karate':
        print('is reading {} dataset...'.format(dataset_name))
        data = tg.datasets.KarateClub(transform=T.NormalizeFeatures())[0]
        if not use_features:
            feature = np.identity(data.num_nodes)
            data.x = torch.Tensor(feature).float()

        ls_data = [data]

    # release gpu memory
    torch.cuda.empty_cache()
    return ls_data