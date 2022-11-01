import torch
from torch_geometric.datasets import TUDataset
import os.path as osp
from gmixup import prepare_dataset_onehot_y
from utils import split_class_graphs
from graphon_estimator import universal_svd, largest_gap
from utils import align_graphs, stat_graph
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

dataset_names = ['IMDB-BINARY', 'REDDIT-BINARY', 'IMDB-MULTI']
# graphon_sizes = [17, 15, 12]
data_path = './'
align_max_size = 1000

for dataset_name in dataset_names:
    path = osp.join(data_path, dataset_name)
    dataset = TUDataset(path, name=dataset_name)
    dataset = list(dataset)
    print('---- WORKING WITH DATASET ' + dataset_name + ' ------')

    # TODO what does this line do? I guess it assigns a class to each graph (comment by Vuk on 1.11.2022)
    for graph in dataset:
        graph.y = graph.y.view(-1)

    dataset = prepare_dataset_onehot_y(dataset)
    random.seed(1314)
    random.shuffle(dataset)
    avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(
        dataset)
    print('Median num nodes: ', int(median_num_nodes))
    print('Avg num nodes: ', int(avg_num_nodes))
    graphon_size = int(avg_num_nodes)
    # TODO what does this split_class_graphs method does
    class_graphs = split_class_graphs(dataset)
    print('Finished splitting class graphs')
    fig, ax = plt.subplots(1, len(class_graphs), figsize=(2 * len(class_graphs), 2), facecolor='w')
    if dataset_name == 'IMDB-MULTI':
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    graphons = []
    for label, graphs in class_graphs:
        align_graphs_list, normalized_node_degrees, max_num, min_num = align_graphs(
            graphs[:align_max_size], padding=True, N=int(graphon_size))
        print('Finished aligning graphs of label ', label)
        graphon = largest_gap(align_graphs_list, k=graphon_size)
        np.fill_diagonal(graphon, 0)  # Confirmed in correspondence with the author
        graphons.append((label, graphon))

    for (label, graphon), axis, i in zip(graphons, ax, range(len(graphons))):
        print(f"graphon info: label:{label}; mean: {graphon.mean()}, shape, {graphon.shape}")
        im = axis.imshow(graphon, vmin=0, vmax=1, cmap=plt.cm.plasma)
        axis.set_title(f"Class {i}", weight="bold")
        axis.axis('off')
    if dataset_name == 'IMDB-MULTI':
        fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    fig.suptitle(dataset_name, y=0.1, weight="bold")
    plt.savefig(f'../fig/{dataset_name}.png', facecolor='white', bbox_inches='tight')