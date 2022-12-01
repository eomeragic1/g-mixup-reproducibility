import networkx as nx
from torch_geometric.datasets import TUDataset
import os.path as osp
from gmixup import prepare_dataset_onehot_y
from utils import split_class_graphs
from graphon_estimator import universal_svd, largest_gap
from utils import align_graphs, stat_graph
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import two_graphons_mixup
from torch_geometric.utils import to_dense_adj

def load_dataset(dataset_name = 'REDDIT-BINARY'):
    data_path = './'
    lam_range = [0.005, 0.01]
    seed = 45
    random.seed(seed)

    # Preparing the dataset
    path = osp.join(data_path, dataset_name)
    dataset = TUDataset(path, name=dataset_name)
    dataset = list(dataset)

    return dataset

def experiment_2(dataset_name = 'REDDIT-BINARY', align_max_size = 500):
    dataset = load_dataset(dataset_name)
    train_nums = int(len(dataset) * 0.7)

    # To each graph we will assign its label, so that is more easily accessed.
    for graph in dataset:
        # graph.y holds a tensor object, and we want it to be a simple value, for example integer 0 or 1, string 'T' or 'F'...
        graph.y = graph.y.view(-1)

    dataset = prepare_dataset_onehot_y(dataset)
    random.shuffle(dataset)
    avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(
        dataset)
    print('Median num nodes: ', int(median_num_nodes))
    class_graphs = split_class_graphs(dataset)
    print('Finished splitting class graphs')

    graphon_size = 50

    graphons = []
    for i, (label, graphs) in enumerate(class_graphs):
        align_graphs_list, normalized_node_degrees, max_num, min_num, sum_graph = align_graphs(graphs[:align_max_size], padding=True, N=int(graphon_size))
        print('Finished aligning graphs of label ', label)
        graphon = largest_gap(align_graphs_list, k=graphon_size, sum_graph=sum_graph)
        np.fill_diagonal(graphon, 0)  # Confirmed in correspondence with the author
        graphons.append((label, graphon))

        # This will draw random graphs from each class
        fig, ax = plt.subplots(1, 4, figsize=(12, 3), facecolor='w')
    #     for axis, graph in zip(ax, random.sample([graph for graph in graphs if graph.shape[0]<80], 4)):
        for axis, graph in zip(ax, random.sample([graph for graph in graphs if graph.num_nodes<80], 4)):
            #         g = nx.from_numpy_matrix(graph) # get adjacency matrix from graph
            g = nx.from_edgelist(graph.edge_index.T.tolist())
            nx.draw_spring(g, ax=axis, node_size=20)
        fig.suptitle(f'Graphs of label {label}')
        plt.savefig(f'../fig/graphs_reddit_{i}.png', facecolor='white', bbox_inches='tight')
        plt.show()


    # graphons will be of length 2 (number of classes)
    # This will draw graphons for each class
    draw_graphons(dataset_name, graphons)


    # Now, let's mix the graphons, and draw the synthetics graphs from the mixed graphon
    num_sample = 3 # we will take 3 graphs from each class
    new_graphs_10 = []
    new_graphs_01 = []
    new_graphs_0505 = []
    two_graphons = random.sample(graphons, 2)
    new_graphs_10 += two_graphons_mixup(two_graphons, la=1, num_sample=num_sample)
    new_graphs_01 += two_graphons_mixup(two_graphons, la=0, num_sample=num_sample)
    new_graphs_0505 += two_graphons_mixup(two_graphons, la=0.5, num_sample=num_sample)

    for i, graph_type in enumerate((new_graphs_10, new_graphs_01, new_graphs_0505)):
        fig, ax = plt.subplots(1, 3, figsize=(9, 3), facecolor='w')
        for axis, graph in zip(ax, graph_type):
            g = nx.from_numpy_matrix(to_dense_adj(graph.edge_index)[0].numpy())
            nx.draw_spring(g, ax=axis, node_size=20)
        plt.suptitle(f'Graphs of label {graph_type[0].y.numpy()}')
        plt.savefig(f'../fig/graphs_reddit_new_{i}.png', facecolor='white', bbox_inches='tight')
        plt.show()

def draw_graphons(dataset_name, graphons, save_fig=False, fig_name=''):
    fig, ax = plt.subplots(1, len(graphons), figsize=(6, 3), facecolor='w')
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    for (label, graphon), axis, i in zip(graphons, ax, range(len(graphons))):
        print(f"graphon info: label:{label}; mean: {graphon.mean()}, shape, {graphon.shape}")
        im = axis.imshow(graphon, vmin=0, vmax=1, cmap=plt.cm.plasma)
        axis.set_title(f"Class {i}", weight="bold")
        axis.axis('off')

    fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    fig.suptitle(dataset_name, y=0.1, weight="bold")

    if save_fig:
        plt.savefig(f'../fig/{fig_name}.png', facecolor='white', bbox_inches='tight')
        plt.show()

def logical_test_experiment_2(dataset_name = 'REDDIT-BINARY', align_max_size = 500):
    """
    Point of this experiment is to create 500 synthethic graphs from the mixed graphon,
    and then estimate a graphon on the synthetic graphs to see is it really a combination of the original graphons
    """
    dataset = load_dataset(dataset_name)
    train_nums = int(len(dataset) * 0.7)

    # To each graph we will assign its label, so that is more easily accessed.
    for graph in dataset:
        # graph.y holds a tensor object, and we want it to be a simple value, for example integer 0 or 1, string 'T' or 'F'...
        graph.y = graph.y.view(-1)

    dataset = prepare_dataset_onehot_y(dataset)
    random.shuffle(dataset)
    avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(
        dataset)
    print('Median num nodes: ', int(median_num_nodes))
    class_graphs = split_class_graphs(dataset)
    print('Finished splitting class graphs')

    graphon_size = 15

    graphons = []
    for i, (label, graphs) in enumerate(class_graphs):
        align_graphs_list, normalized_node_degrees, max_num, min_num, sum_graph = align_graphs(graphs[:align_max_size],
                                                                                               padding=True,
                                                                                               N=int(graphon_size))
        print('Finished aligning graphs of label ', label)
        graphon = largest_gap(align_graphs_list, k=graphon_size, sum_graph=sum_graph)
        np.fill_diagonal(graphon, 0)  # Confirmed in correspondence with the author
        graphons.append((label, graphon))

    num_sample = 500
    new_graphs_10 = []
    new_graphs_01 = []
    new_graphs_0505 = []
    two_graphons = random.sample(graphons, 2)
    new_graphs_0505 += two_graphons_mixup(two_graphons, la=0.5, num_sample=num_sample)

    align_graphs_list, normalized_node_degrees, max_num, min_num, sum_graph = align_graphs(new_graphs_0505,
                                                                                           padding=True,
                                                                                           N=int(graphon_size))
    print('Finished aligning graphs of label ', 'mixed')
    mixed_graphon = largest_gap(align_graphs_list, k=graphon_size, sum_graph=sum_graph)
    np.fill_diagonal(graphon, 0)

    graphons.append(('mixed', mixed_graphon))
    draw_graphons(dataset_name, graphons, save_fig=True, fig_name='mixed_graphon_experiment')





# graphon_size = int(median_num_nodes)

# experiment_2()
logical_test_experiment_2()