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

def experiment_1(dataset_names: list, graphon_sizes: list, data_path = './', align_max_size = 1000):
    for dataset_name, graphon_size in zip(dataset_names, graphon_sizes):
        path = osp.join(data_path, dataset_name)
        dataset = TUDataset(path, name=dataset_name)
        dataset = list(dataset)
        a = 'control-point 1'

        # To each graph we will assign its label, so that is more easily accessed.
        for graph in dataset:
            # graph.y holds a tensor object, and we want it to be a simple value, for example integer 0 or 1, string 'T' or 'F'...
            graph.y = graph.y.view(-1)

        dataset = prepare_dataset_onehot_y(dataset)
        random.seed(1314)
        random.shuffle(dataset)
        avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(
            dataset)
        print('Median num nodes: ', int(median_num_nodes))
        print('Avg num nodes: ', int(avg_num_nodes))
        # method split_class_graphs outputs a list of lists, where each inner list contains
        # dense adjacency matrices of graphs of same class
        # Example : output = [[g1_label0,g2_label0,g3_label0,...], [g1_label1,g2_label1,...],...]
        class_graphs = split_class_graphs(dataset)
        print('Finished splitting class graphs')
        # arguments for plotting are set so that we have enough space for graphon of each class to be shown
        fig, ax = plt.subplots(1, len(class_graphs), figsize=(2 * len(class_graphs), 2), facecolor='w')

        # additional alignment for IMDB-MULTI dataset
        if dataset_name == 'IMDB-MULTI':
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
        graphons = []
        # In following lines Graphon estimation is done for each class in the current dataset!
        for label, graphs in class_graphs:
            align_graphs_list, normalized_node_degrees, max_num, min_num, sum_graph = align_graphs(
                graphs[:align_max_size], padding=True, N=int(graphon_size))
            print('Finished aligning graphs of label ', label)
            graphon = largest_gap(align_graphs_list, k=graphon_size, sum_graph=sum_graph)
            np.fill_diagonal(graphon, 0)  # Confirmed in correspondence with the author
            graphons.append((label, graphon))

        # Plotting of graphons for current dataset
        for (label, graphon), axis, i in zip(graphons, ax, range(len(graphons))):
            print(f"graphon info: label:{label}; mean: {graphon.mean()}, shape, {graphon.shape}")
            im = axis.imshow(graphon, vmin=0, vmax=1, cmap=plt.cm.plasma)
            axis.set_title(f"Class {i}", weight="bold")
            axis.axis('off')
        if dataset_name == 'IMDB-MULTI':
            fig.colorbar(im, cax=cbar_ax, orientation='vertical')
        fig.suptitle(dataset_name, y=0.1, weight="bold")
        plt.savefig(f'../fig/{dataset_name}.png', facecolor='white', bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    dataset_names = ['IMDB-BINARY', 'REDDIT-BINARY', 'IMDB-MULTI']
    # Graphon sizes are set to match the Figure 2 in paper, the actual average number of nodes in graph sets is different
    # for REDDIT-BINARY and IMDB-MULTI
    graphon_sizes_from_figure = [19, 15, 12]
    graphon_sizes_from_average_num_of_nodes = [19, 429, 13]

    experiment_1(dataset_names, graphon_sizes_from_figure)
    # experiment_1(dataset_names, graphon_sizes_from_average_num_of_nodes)