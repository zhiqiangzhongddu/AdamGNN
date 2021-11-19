import numpy as np
import torch
import torch_geometric as tg


def prepare_NC_data(data, batch_size, dense, device, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if batch_size is not None:
        num_batch = int(np.ceil(data.num_nodes / batch_size))
    else:
        num_batch = 1
    batch = torch.cat(
        [torch.FloatTensor([n] * int(np.ceil(data.num_nodes / num_batch) + 1)) for n in range(num_batch)]
    ).long()[:data.num_nodes]

    if dense:
        edge_matrix = tg.utils.to_dense_adj(data.edge_index, batch=None)[0]
        data.edge_matrix = edge_matrix
    data.batch = batch
    data.edge_weight = torch.ones(data.num_edges)
    data.to(device)

    return data


def prepare_LP_data_pyg(data, batch_size, device):
    edge_matrix = tg.utils.to_dense_adj(data.train_pos_edge_index, batch=None)[0].to(device)

    if batch_size is not None:
        num_batch = int(np.ceil(data.num_nodes / batch_size))
    else:
        num_batch = 1
    batch = torch.cat(
        [torch.FloatTensor([n] * int(np.ceil(data.num_nodes / num_batch) + 1)) for n in range(num_batch)]
    )[:data.num_nodes]
    data.edge_matrix = edge_matrix
    data.batch = batch
    data.edge_weight = torch.ones(data.train_pos_edge_index.shape[1])
    data.train_neg_adj_mask = None

    return data


def prepare_LP_data(data, df_train, df_valid, df_test, batch_size, device):

    data.train_mask = data.val_mask = data.test_mask = None

    df_train_pos = df_train[df_train['label'] == 1]
    df_train_neg = df_train[df_train['label'] == 0]
    df_val_pos = df_valid[df_valid['label'] == 1]
    df_val_neg = df_valid[df_valid['label'] == 0]
    df_test_pos = df_test[df_test['label'] == 1]
    df_test_neg = df_test[df_test['label'] == 0]

    train_pos_edge_index = torch.tensor(df_train_pos[['source', 'target']].values).t()
    train_neg_edge_index = torch.tensor(df_train_neg[['source', 'target']].values).t()
    val_pos_edge_index = torch.tensor(df_val_pos[['source', 'target']].values).t()
    val_neg_edge_index = torch.tensor(df_val_neg[['source', 'target']].values).t()
    test_pos_edge_index = torch.tensor(df_test_pos[['source', 'target']].values).t()
    test_neg_edge_index = torch.tensor(df_test_neg[['source', 'target']].values).t()

    if batch_size is not None:
        num_batch = int(np.ceil(data.num_nodes / batch_size))
    else:
        num_batch = 1
    batch = torch.cat(
        [torch.FloatTensor([n] * int(np.ceil(data.num_nodes / num_batch) + 1)) for n in range(num_batch)]
    )[:data.num_nodes]

    data.train_pos_edge_index = train_pos_edge_index
    data.train_neg_edge_index = train_neg_edge_index
    data.val_pos_edge_index = val_pos_edge_index
    data.val_neg_edge_index = val_neg_edge_index
    data.test_pos_edge_index = test_pos_edge_index
    data.test_neg_edge_index = test_neg_edge_index

    data.batch = batch
    data.edge_index = data.train_pos_edge_index
    data.edge_index = tg.utils.to_undirected(data.edge_index)
    data.edge_index = tg.utils.remove_self_loops(data.edge_index)[0]
    # data.train_pos_edge_index = tg.utils.add_self_loops(data.train_pos_edge_index)[0]
    data.edge_weight = torch.ones(data.edge_index.shape[1])
    data.to(device)

    print('train pos {}, train neg {} valid pos {}, valid neg {}, test pos {}, test neg {}'.format(
        data.train_pos_edge_index.shape[1], data.train_neg_edge_index.shape[1],
        data.val_pos_edge_index.shape[1], data.val_neg_edge_index.shape[1],
        data.test_pos_edge_index.shape[1], data.test_neg_edge_index.shape[1]
    ))

    return data
