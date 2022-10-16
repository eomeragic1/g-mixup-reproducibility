# Some code taken from:
# https://github.com/Shen-Lab/GraphCL/blob/master/unsupervised_Cora_Citeseer/aug.py
import copy
import random
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import dropout, subgraph
import torch_geometric.data
import torch
from torch_geometric.utils.num_nodes import maybe_num_nodes


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

def augment_dataset_dropedge(loader: torch_geometric.data.DataLoader, aug_percent=0.2):
    idx = random.sample(range(len(loader.dataset)), k=int(aug_percent*len(loader.dataset)))
    for index in idx:
        loader.dataset[index].edge_index, _ = dropout_edge(loader.dataset[index].edge_index, p=0.1,
                                                           force_undirected=True)
    return loader

def augment_dataset_dropnode(loader: torch_geometric.data.DataLoader, aug_percent=0.2):
    idx = random.sample(range(len(loader.dataset)), k=int(aug_percent * len(loader.dataset)))
    for index in idx:
        new_edge_index, _, new_node_mask = dropout_node(loader.dataset[index].edge_index, p=0.5,
                                                           num_nodes=loader.dataset[index].num_nodes)
        loader.dataset[index] = Data(x=loader.dataset[index].x[new_node_mask, :], edge_index=new_edge_index,
                                     y=loader.dataset[index].y)
    return loader

