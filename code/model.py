import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, JumpingKnowledge, GAE, InnerProductDecoder
from torch_scatter import scatter
from torch_sparse import coalesce
from torch_sparse import transpose
from torch_sparse import spspmm

from utils import kl_loss, recon_loss

import warnings

warnings.filterwarnings('ignore')


# %matplotlib inline


def StAS(index_A, value_A, index_S, value_S, device, N, kN):
    r"""StAS: a function which returns new edge weights for the pooled graph using the formula S^{T}AS"""

    index_A, value_A = coalesce(index_A, value_A, m=N, n=N)
    index_S, value_S = coalesce(index_S, value_S, m=N, n=kN)
    index_B, value_B = spspmm(index_A, value_A, index_S, value_S, N, N, kN)

    index_St, value_St = transpose(index_S, value_S, N, kN)
    index_B, value_B = coalesce(index_B, value_B, m=N, n=kN)
    index_E, value_E = spspmm(index_St, value_St, index_B, value_B, kN, N, kN)

    return index_E, value_E


def graph_connectivity(x, cluster_ids, edge_index, edge_weight, cluster_index, score, N, do_view, device):
    r"""graph_connectivity: is a function which internally calls StAS func to maintain graph connectivity"""

    # nodes included in pooled ego networks
    mask_in = (cluster_index[1].unsqueeze(-1) == cluster_ids).any(-1).nonzero().squeeze()
    index_in = cluster_index[:, mask_in]
    value_in = score[mask_in]
    # nodes do not included in any pooled ego networks, we treat them as nodes in pooled graph
    # Complement set operation
    # non_in = cluster_index.unique()[((cluster_index.unique().unsqueeze(-1)==index_in.unique()).sum(-1)==0)]
    # non_in = torch.LongTensor(list(set(cluster_index.unique().tolist())-set(index_in.unique().tolist()))).to(device)
    # non_in = torch.LongTensor(list(set(edge_index.unique().tolist())-set(index_in.unique().tolist()))).to(device)
    non_in = torch.LongTensor(list(set(range(N)) - set(index_in.unique().tolist()))).to(device)
    # creat the S
    index_S = torch.cat(
        [index_in, torch.stack([cluster_ids, cluster_ids], axis=0), torch.stack([non_in, non_in], axis=0)], axis=-1)
    value_S = F.pad(input=value_in, pad=(0, index_S.shape[1] - value_in.shape[0]), mode='constant', value=1.)
    # # sort
    # _, indices = torch.sort(index_S, 1)
    # index_S = index_S[:, indices[1]]
    # value_S = value_S[indices[1]]

    # calculate the number of nodes in the pooled graph
    kN = index_S[1].unique().size(0)
    # update cluster_ids with remaining nodes
    ego_ids = index_S[1].unique()
    if do_view:
        print(f'Origin graph has {N} nodes, {N - cluster_index.unique().size(0)} isolated nodes')
        print(f'Pooled graph has {kN} nodes, {cluster_ids.size(0)} ego nodes, {non_in.size(0)} remaining nodes')

    # relabel for pooling ie: make S [N x kN]
    n_idx = torch.zeros(N, dtype=torch.long)
    n_idx[ego_ids] = torch.arange(ego_ids.size(0))
    index_S[1] = n_idx[index_S[1]]

    # generate ego feature
    ego_x = x[ego_ids]

    # create A
    index_A = edge_index.clone()
    if edge_weight is None:
        value_A = value_S.new_ones(edge_index[0].size(0))
    else:
        value_A = edge_weight.clone()

    index_E, value_E = StAS(index_A, value_A, index_S, value_S, device, N, kN)
    index_E, value_E = tg.utils.add_remaining_self_loops(edge_index=index_E, edge_weight=value_E, fill_value=1.)

    return index_E, value_E, index_S, value_S, ego_x


def Generate_high_order_adj(edge_index, order, M):
    res_index = edge_index.clone()

    for _ in range(order - 1):
        res_index, _ = spspmm(res_index, None, edge_index, None, M, M, M, coalesced=True)

    return res_index


def Generate_clusters(edge_index, num_hops):
    M = edge_index[0].max().item() + 1
    # edge_index, _ = tg.utils.remove_self_loops(edge_index=edge_index)

    # For directed graph
    ls_index = [edge_index]
    for order in range(2, num_hops + 1):
        ls_index.append(Generate_high_order_adj(
            edge_index=edge_index,
            order=order,
            M=M
        ))
    cluster_index = torch.cat(ls_index, axis=-1)
    cluster_index = cluster_index.unique(dim=-1)
    # # For undirected graph
    # cluster_index = Generate_high_order_adj(
    #     edge_index=edge_index,
    #     order=num_hops,
    #     M=M
    # )
    cluster_index, _ = tg.utils.remove_self_loops(edge_index=cluster_index)
    return cluster_index


def Cluster_assignment(fitness, index):
    out = scatter(fitness, index[1], dim=0, reduce='mean')
    return out


def Select_clusters(edge_index, scores, range, overlap, N, device):
    if overlap:
        index, _ = tg.utils.remove_self_loops(edge_index=edge_index)
        all_idx = index[0].unique()
        del_idx = index[0][(scores[index[0]] - scores[index[1]]) < 0].unique()
        # Complement set operation
        # cluster_ids = all_idx[((all_idx.unsqueeze(-1)==del_idx).sum(-1)==0)]
        cluster_ids = torch.LongTensor(list(set(all_idx.tolist()) - set(del_idx.tolist()))).to(device)
    else:
        index = Generate_clusters(edge_index=edge_index,
                                  num_hops=range * 2)
        all_idx = index[0].unique()
        del_idx = index[0][(scores[index[0]] - scores[index[1]]) < 0].unique()
        # Complement set operation
        # cluster_ids = all_idx[((all_idx.unsqueeze(-1)==del_idx).sum(-1)==0)]
        cluster_ids = torch.LongTensor(list(set(all_idx.tolist()) - set(del_idx.tolist()))).to(device)
    return cluster_ids


class Merge_xs(nn.Module):
    def __init__(self, args, mode, dim, num_levels):
        super(Merge_xs, self).__init__()
        self.args = args
        self.mode = mode
        self.dim = dim
        self.num_levels = num_levels
        self.drop_ratio = args.drop_ratio
        self.device = args.device

        if self.num_levels > 1:
            if self.mode == 'MAX':
                self.out_cat = JumpingKnowledge(mode='max', channels=self.dim, num_layers=self.num_levels)
            elif self.mode == 'LSTM':
                self.out_cat = JumpingKnowledge(mode='lstm', channels=self.dim, num_layers=self.num_levels)
            elif self.mode == 'ATT':
                self.lin_att = nn.Linear(2 * self.dim, 1)

    def forward(self, xs):
        score = None

        if self.mode == 'NONE':
            embedding = xs[0]
        elif self.mode == 'MEAN':
            # mean
            embedding = torch.mean(torch.stack(xs), dim=0)
        elif self.mode == 'LINEAR':
            # linear transfer
            embedding = self.out_cat(xs=xs)
            embedding = self.lin_top_down(embedding)
        elif self.mode == 'ATT':
            query = xs[0]
            message = torch.cat(xs[1:], axis=0)
            N = query.shape[0]
            # normalize inputs
            query = F.normalize(query, p=2, dim=-1)
            message = F.normalize(message, p=2, dim=-1)
            score = self.lin_att(torch.cat((message, query.repeat(self.num_levels - 1, 1)), dim=-1)).squeeze(-1)
            score = F.leaky_relu(score, inplace=False)
            # sparse softmax
            index = torch.LongTensor(
                [list(range(N, N * (self.num_levels))), list(range(N)) * (self.num_levels - 1)]
            ).to(self.device)
            score = tg.utils.softmax(score, index[1], num_nodes=N * self.num_levels)
            # Sample attention coefficients stochastically.
            score = F.dropout(score, p=self.drop_ratio, training=self.training)
            # add weight to message
            message = score.unsqueeze(-1) * message
            # obtain final embedding
            embedding = query + scatter(message, index[1], dim=0, reduce='add')
        else:
            embedding = self.out_cat(xs=xs)

        return embedding, score


class Encoder(nn.Module):
    def __init__(self, feat_dim, hid_dim, agg_gnn, drop_ratio, num_levels, encoder_layers):
        super(Encoder, self).__init__()
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.agg_gnn = agg_gnn
        self.drop_ratio = drop_ratio
        self.num_levels = num_levels
        self.encoder_layers = encoder_layers

        self.convs = nn.ModuleList()
        for level in range(self.num_levels):
            level_convs = nn.ModuleList()
            for layer in range(self.encoder_layers):
                if (level == 0) & (layer == 0):
                    if self.agg_gnn == 'GCN':
                        level_convs.append(GCNConv(self.feat_dim, self.hid_dim))
                    elif self.agg_gnn == 'SAGE':
                        level_convs.append(SAGEConv(self.feat_dim, self.hid_dim))
                    elif self.agg_gnn == 'GAT':
                        level_convs.append(GATConv(
                            self.feat_dim, self.hid_dim // 8,
                            heads=8, dropout=self.drop_ratio
                        ))
                else:
                    if self.agg_gnn == 'GCN':
                        level_convs.append(GCNConv(self.hid_dim, self.hid_dim))
                    elif self.agg_gnn == 'SAGE':
                        level_convs.append(SAGEConv(self.hid_dim, self.hid_dim))
                    elif self.agg_gnn == 'GAT':
                        level_convs.append(GATConv(
                            self.hid_dim, self.hid_dim // 8,
                            heads=8, dropout=self.drop_ratio
                        ))
            self.convs.append(level_convs)

    def forward(self, x, level, edge_index, edge_weight):
        level_convs = self.convs[level]
        if self.agg_gnn == 'GCN':
            for conv in level_convs[:-1]:
                x = conv(x, edge_index, edge_weight)
                x = F.relu(x)
                x = F.dropout(x, p=self.drop_ratio, training=self.training)
            x = level_convs[-1](x, edge_index, edge_weight)
        else:
            for conv in level_convs[:-1]:
                x = conv(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.drop_ratio, training=self.training)
            x = level_convs[-1](x, edge_index)
        return x


class Adaptive_pooling(nn.Module):
    def __init__(self, args, in_size, cluster_range):
        super(Adaptive_pooling, self).__init__()
        self.args = args
        self.in_size = in_size
        self.cluster_range = cluster_range
        self.fitness_mode = args.fitness_mode
        self.pooling_mode = args.pooling_mode
        self.do_view = args.do_view
        self.overlap = args.overlap
        self.drop_ratio = args.drop_ratio
        self.device = args.device

        self.score_lin = nn.Linear(2 * self.in_size, 1)
        self.pool_lin = nn.Linear(2 * self.in_size, 1)

    def calculate_fitness(self, x, index, N, batch=None):
        # all values in [0, 1] and disgnal values are 1
        if self.fitness_mode == 'c' or 'both' in self.fitness_mode:
            # linear scores
            x = F.normalize(x, p=2, dim=-1)
            lin_score = torch.sum(x[index[0]] * x[index[1]], dim=-1)
            lin_score = tg.utils.softmax(lin_score, index[0], num_nodes=N)
        if self.fitness_mode == 's' or 'both' in self.fitness_mode:
            # non-linear scores
            x_score = self.score_lin(torch.cat((x[index[0]], x[index[1]]), dim=-1)).squeeze(-1)
            x_score = F.leaky_relu(x_score)
            nonlin_score = tg.utils.softmax(x_score, index[0], num_nodes=N)

        if self.fitness_mode == 'both_c':
            fitness = lin_score * nonlin_score
            fitness = tg.utils.softmax(fitness, index[0], num_nodes=N)
        elif self.fitness_mode == 'both_j':
            fitness = lin_score + nonlin_score
            fitness = tg.utils.softmax(fitness, index[0], num_nodes=N)
        elif self.fitness_mode == 'c':
            fitness = lin_score
        elif self.fitness_mode == 's':
            fitness = nonlin_score
        return fitness

    def Pool_nodes(self, x, ego_x, index_S, value_S, N):
        if self.pooling_mode == 'mean':
            # use fitness scores to pool nodes
            x_j = x[index_S[0]]
            _x = scatter(x_j, index_S[1], dim=0, reduce='mean')[index_S[1].unique()]
        elif self.pooling_mode == 'max':
            x_j = x[index_S[0]]
            _x = scatter(x_j, index_S[1], dim=0, reduce='max')[index_S[1].unique()]
        elif self.pooling_mode == 'main':
            # adopt ego center node's feature as pooled node's feature
            _x = ego_x.clone()
        elif self.pooling_mode == 'att':
            # set up query and message
            x_score = self.pool_lin(torch.cat((x[index_S[0]], ego_x[index_S[1]]), dim=-1)).squeeze(-1)
            x_score = F.leaky_relu(x_score)
            x_score = tg.utils.softmax(x_score, index_S[1], num_nodes=N)
            # Sample attention coefficients stochastically
            x_score = F.dropout(x_score, p=self.drop_ratio, training=self.training)
            # learn pooled node features
            x_j = x[index_S[0]] * x_score.unsqueeze(-1)
            x_j = x_j * value_S.unsqueeze(-1)
            _x = scatter(x_j, index_S[1], dim=0, reduce='add')[index_S[1].unique()]
        return _x

    def forward(self, embedding, edge_index, edge_weight):
        N = embedding.size(0)  # number of nodes

        # set up ego networks with lamda hops
        cluster_index = Generate_clusters(
            edge_index=edge_index,
            num_hops=self.cluster_range
        )
        # calculate fitness scores each ego
        fitness = self.calculate_fitness(
            embedding, cluster_index, N
        )
        # calculate the concentration of each cluster
        cluster_scores = Cluster_assignment(
            fitness=fitness,
            index=cluster_index
        )

        # if self.do_view:
        #     print('Fitness min {:.3f}, max {:.3f}, mean {:.3f}'.format(
        #         fitness.min().item(), fitness.max().item(), fitness.mean().item()
        #     ))
        #     print('Cluster score min {:.3f}, max {:.3f}, mean {:.3f}'.format(
        #         cluster_scores.min().item(), cluster_scores.max().item(), cluster_scores.mean().item()
        #     ))

        # select clusters
        cluster_ids = Select_clusters(
            edge_index=edge_index,
            scores=cluster_scores,
            range=self.cluster_range,
            overlap=self.overlap,
            N=N,
            device=self.device
        )
        _edge_index, _edge_weight, index_S, value_S, ego_x = graph_connectivity(
            x=embedding,
            cluster_ids=cluster_ids,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cluster_index=cluster_index,
            score=fitness,
            N=N,
            do_view=self.do_view,
            device=self.device
        )
        _embedding = self.Pool_nodes(
            x=embedding, ego_x=ego_x,
            index_S=index_S, value_S=value_S, N=N
        )
        return _embedding, _edge_index, _edge_weight, index_S, value_S, fitness


class AdamGNN(nn.Module):
    def __init__(self, args, feat_dim, out_dim):
        super(AdamGNN, self).__init__()
        self.args = args
        self.feat_dim = feat_dim
        self.agg_gnn = args.local_agg_gnn
        self.hid_dim = args.hid_dim
        self.drop_ratio = args.drop_ratio
        self.out_dim = out_dim
        self.num_levels = args.num_levels
        self.cluster_range = args.cluster_range
        self.encoder_layers = args.encoder_layers
        self.output_mode = args.output_mode

        # define encoder for each level
        self.encoder = Encoder(
            feat_dim=self.feat_dim, hid_dim=self.hid_dim,
            agg_gnn=self.agg_gnn, drop_ratio=self.drop_ratio,
            num_levels=self.num_levels, encoder_layers=self.encoder_layers
        )
        # define pools
        self.pools = nn.ModuleList()
        for _ in range(self.num_levels - 1):
            self.pools.append(Adaptive_pooling(
                args=self.args, in_size=self.hid_dim, cluster_range=self.cluster_range
            ))
        # define flyback aggregate
        self.out_cat = Merge_xs(
            args=self.args, mode=self.output_mode,
            dim=self.hid_dim, num_levels=self.num_levels
        )
        # define last layer to update node embedding with multi-grained semantics
        self.last_layer = nn.Linear(self.hid_dim, self.out_dim)
        # self.last_layer = nn.Sequential(nn.Linear(self.hid_dim, 32), nn.ReLU(),
        #                     nn.Linear(32, self.out_dim))

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        orig_edge_index = edge_index

        generated_embeddings = []
        index_Bs = []
        value_Bs = []
        ls_index = []

        for level in range(self.num_levels):
            # gnn embedding
            print(f'Level {level} has {edge_index.shape[1]} edges.')
            embedding = self.encoder(x, level, edge_index, edge_weight)
            if level == 0:
                embedding_gnn = embedding.clone()
            if level < (self.num_levels - 1):
                _input, _index, _weight, index_B, value_B, _ = self.pools[level](
                    embedding=embedding,
                    edge_index=edge_index,
                    edge_weight=edge_weight
                )
                _input = F.normalize(_input, p=2, dim=1)
                index_Bs.append(index_B)
                value_Bs.append(value_B)
                ls_index.append(_index.data.cpu().numpy())

                # assign the embedding as next level's input feature matrix
                x = _input
                edge_index = _index
                edge_weight = _weight

            if (level > 0) & (level < (self.num_levels - 1)):
                for index, value in zip(reversed(index_Bs[:-1]), reversed(value_Bs[:-1])):
                    print('learned embedding shape:{}, transform from {} to {}'.format(
                        embedding.shape[0], index[1].max().item() + 1, index[0].max().item() + 1
                    ))
                    embedding = scatter(embedding[index[1]] * value.unsqueeze(-1), index[0], dim=0, reduce='mean')
                    print('after, embedding shape: {}'.format(embedding.shape[0]))
            if level == (self.num_levels - 1):
                for index, value in zip(reversed(index_Bs), reversed(value_Bs)):
                    print('learned embedding shape:{}, transform from {} to {}'.format(
                        embedding.shape[0], index[1].max().item() + 1, index[0].max().item() + 1
                    ))
                    embedding = scatter(embedding[index[1]] * value.unsqueeze(-1), index[0], dim=0, reduce='mean')
                    print('after, embedding shape: {}'.format(embedding.shape[0]))
            generated_embeddings.append(embedding)

        if len(generated_embeddings) > 1:
            embedding, _ = self.out_cat(xs=generated_embeddings)

        loss_kl = kl_loss(mu=embedding, logvar=embedding_gnn)
        loss_recon = recon_loss(z=embedding, pos_edge_index=orig_edge_index)

        embedding = self.last_layer(embedding)

        return embedding, loss_recon, loss_kl, (index_Bs, value_Bs), ls_index
