# Some code taken from:
# https://github.com/Shen-Lab/GraphCL/blob/master/unsupervised_Cora_Citeseer/aug.py
import copy
import random
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout, subgraph, from_scipy_sparse_matrix, to_dense_adj
import torch_geometric.data
import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes
import scipy.sparse as sp
import numpy as np


def relabel_edges(edge_index):
    sorted_indices = torch.unique(edge_index, sorted=True).tolist()
    mapping = {j: i for i, j in enumerate(sorted_indices)}
    return edge_index.apply_(lambda x: mapping[x])

def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True):

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask

def dropout_node(edge_index, p: float = 0.5,
                 num_nodes = None,
                 training = True):

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask

    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p
    edge_index, edge_attr = subgraph(node_mask, edge_index,
                                        num_nodes=num_nodes,
                                        return_edge_mask=False)
    return edge_index, edge_attr, node_mask

def augment_dataset_dropedge(loader: torch_geometric.data.DataLoader, aug_percent=0.2, edge_percent = 0.1):
    idx = random.sample(range(len(loader.dataset)), k=int(aug_percent*len(loader.dataset)))
    for index in idx:
        loader.dataset[index].edge_index, _ = dropout_edge(loader.dataset[index].edge_index, p=edge_percent,
                                                           force_undirected=True)
    return DataLoader(loader.dataset, batch_size=128, shuffle=True)

def augment_dataset_dropnode(loader: torch_geometric.data.DataLoader, aug_percent=0.2, node_percent=0.1):
    idx = random.sample(range(len(loader.dataset)), k=int(aug_percent * len(loader.dataset)))
    for index in idx:
        new_edge_index, _, new_node_mask = dropout_node(loader.dataset[index].edge_index, p=node_percent,
                                                           num_nodes=loader.dataset[index].num_nodes)
        num_nodes = torch.sum(new_node_mask).item()

        # Could happen that all nodes are removed which creates batching problems, that's why we only keep the augs
        # with at least one node
        if num_nodes != 0:
            loader.dataset[index] = Data(edge_index=relabel_edges(new_edge_index), y=loader.dataset[index].y,
                                     x=loader.dataset[index].x[new_node_mask, :], num_nodes=num_nodes)
    return DataLoader(loader.dataset, batch_size=128, shuffle=True)


def aug_subgraph(input_fea, input_adj, drop_percent=0.2):
    input_adj = to_dense_adj(input_adj).squeeze(0)
    node_num = input_fea.size()[0]

    all_node_list = [i for i in range(node_num)]
    s_node_num = int(node_num * (1 - drop_percent))
    center_node_id = random.randint(0, node_num - 1)
    sub_node_id_list = [center_node_id]
    all_neighbor_list = []

    for i in range(s_node_num - 1):

        all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()

        all_neighbor_list = list(set(all_neighbor_list))
        new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]
            sub_node_id_list.append(new_node)
        else:
            break

    drop_node_list = sorted([i for i in all_node_list if not i in sub_node_id_list])

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea
    aug_input_adj = aug_input_adj.nonzero().t().contiguous()

    return aug_input_fea, aug_input_adj

def delete_row_col(input_matrix, drop_list, only_row=False):

    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out

def augment_dataset_subgraph(loader: torch_geometric.data.DataLoader, aug_percent=0.2):
    idx = random.sample(range(len(loader.dataset)), k=int(aug_percent * len(loader.dataset)))
    for index in idx:
        if loader.dataset[index].edge_index.numel():
            new_x, new_adj = aug_subgraph(loader.dataset[index].x, loader.dataset[index].edge_index, drop_percent=0.1)
            num_nodes = new_x.size()[0]

            # Could happen that all nodes are removed which creates batching problems, that's why we only keep the augs
            # with at least one node
            if num_nodes != 0:
                loader.dataset[index] = Data(edge_index=new_adj, y=loader.dataset[index].y, x=new_x, num_nodes=num_nodes)
    return DataLoader(loader.dataset, batch_size=128, shuffle=True)

def add_random_edge(edge_index, p: float, force_undirected = False,
                    num_nodes=None,
                    training: bool = True):
    if p < 0. or p > 1.:
        raise ValueError(f'Ratio of added edges has to be between 0 and 1 '
                         f'(got {p}')
    if force_undirected and isinstance(num_nodes, (tuple, list)):
        raise RuntimeError('`force_undirected` is not supported for'
                           ' heterogeneous graphs')

    device = edge_index.device
    if not training or p == 0.0:
        edge_index_to_add = torch.tensor([[], []], device=device)
        return edge_index, edge_index_to_add

    if not isinstance(num_nodes, (tuple, list)):
        num_nodes = (num_nodes, num_nodes)
    num_src_nodes = maybe_num_nodes(edge_index, num_nodes[0])
    num_dst_nodes = maybe_num_nodes(edge_index, num_nodes[1])

    num_edges_to_add = round(edge_index.size(1) * p)
    row = torch.randint(0, num_src_nodes, size=(num_edges_to_add, ))
    col = torch.randint(0, num_dst_nodes, size=(num_edges_to_add, ))

    if force_undirected:
        mask = row < col
        row, col = row[mask], col[mask]
        row, col = torch.cat([row, col]), torch.cat([col, row])
    edge_index_to_add = torch.stack([row, col], dim=0).to(device)
    edge_index = torch.cat([edge_index, edge_index_to_add], dim=1)
    return edge_index, edge_index_to_add


def augment_dataset_addedge(loader: torch_geometric.data.DataLoader, aug_percent=1, edge_percent = 0.1):
    idx = random.sample(range(len(loader.dataset)), k=int(aug_percent * len(loader.dataset)))
    for i in idx:
        loader.dataset[i].edge_index, _ = add_random_edge(loader.dataset[i].edge_index, p=edge_percent,
                                                           force_undirected=True, num_nodes=loader.dataset[i].num_nodes)
    return DataLoader(loader.dataset, batch_size=128, shuffle=True)