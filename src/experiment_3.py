import random
import torch
import os.path as osp
from torch_geometric.datasets import TUDataset
from gmixup import prepare_dataset_onehot_y
from utils import stat_graph
import numpy as np
from graphon_estimator import largest_gap
from utils import split_class_graphs, align_graphs
from torch_geometric.loader import DataLoader
from gmixup import prepare_dataset_x
from utils import two_graphons_mixup
from models import GIN, GCN, DiffPoolNet, TopKNet, MinCutPoolNet
from gmixup import mixup_cross_entropy_loss
from augmentations import augment_dataset_dropedge, augment_dataset_dropnode, augment_dataset_subgraph
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, num_classes, optimizer):
    model.train()
    loss_all = 0
    graph_all = 0
    correct = 0
    total = 0
    for data in train_loader:
        # print( "data.y", data.y )
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        y = data.y.view(-1, num_classes)
        #print(y.size())
        #print(output.size())
        loss = mixup_cross_entropy_loss(output, y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        graph_all += data.num_graphs
        optimizer.step()
        y = y.max(dim=1)[1]
        pred = output.max(dim=1)[1]
        correct += pred.eq(y).sum().item()
        total += data.num_graphs

    loss = loss_all / graph_all
    acc = correct / total
    return model, loss, acc

def test(model, loader, num_classes):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        y = data.y.view(-1, num_classes)
        loss += mixup_cross_entropy_loss(output, y).item() * data.num_graphs
        y = y.max(dim=1)[1]
        correct += pred.eq(y).sum().item()
        total += data.num_graphs
    acc = correct / total
    loss = loss / total
    return acc, loss

def run_test(id, dataset_name, model_name, seed, aug):
    start = time.time()
    data_path = './'
    epochs = 300
    batch_size = 128
    lr = 0.01
    num_hidden = 64
    lam_range = [0.1, 0.2]
    aug_ratio = 0.2
    aug_num = 10

    path = osp.join(data_path, dataset_name)
    dataset = TUDataset(path, name=dataset_name)
    dataset = list(dataset)
    for graph in dataset:
        graph.y = graph.y.view(-1)

    dataset = prepare_dataset_onehot_y(dataset)
    train_nums = int(len(dataset) * 0.7)
    train_val_nums = int(len(dataset) * 0.8)

    avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(
        dataset[:train_nums])
    graphon_size = int(median_num_nodes)
    print(f"Avg num nodes of training graphs: {avg_num_nodes}")
    print(f"Avg num edges of training graphs: {avg_num_edges}")
    print(f"Avg density of training graphs: {avg_density}")
    print(f"Median num edges of training graphs: {median_num_edges}")
    print(f"Median density of training graphs: {median_density}")

    torch.manual_seed(seed)
    random.seed(seed)
    random.shuffle(dataset)

    if aug == 'G-Mixup':
        class_graphs = split_class_graphs(dataset[:train_nums])
        graphons = []
        for label, graphs in class_graphs:
            align_graphs_list, normalized_node_degrees, max_num, min_num, sum_graph = align_graphs(
                graphs, padding=True, N=graphon_size)
            graphon = largest_gap(align_graphs_list, k=graphon_size, sum_graph=sum_graph)
            graphons.append((label, graphon))

        num_sample = int(train_nums * aug_ratio / aug_num)
        lam_list = np.random.uniform(low=lam_range[0], high=lam_range[1], size=(aug_num,))

        random.seed(seed)
        new_graph = []
        for lam in lam_list:
            two_graphons = random.sample(graphons, 2)
            new_graph += two_graphons_mixup(two_graphons, la=lam, num_sample=num_sample)

        new_dataset = new_graph + dataset
        new_train_nums = train_nums + len(new_graph)
        new_train_val_nums = train_val_nums + len(new_graph)
    else:
        new_dataset = dataset
        new_train_nums = train_nums
        new_train_val_nums = train_val_nums

    new_dataset = prepare_dataset_x(new_dataset)

    num_features = new_dataset[0].x.shape[1]
    num_classes = new_dataset[0].y.shape[0]

    # avg_num_nodes, avg_num_edges, avg_density, median_num_nodes, median_num_edges, median_density = stat_graph(new_dataset[:new_train_nums])
    # print(f"Avg num nodes of new training graphs: {avg_num_nodes}")
    # print(f"Avg num edges of new training graphs: {avg_num_edges}")
    # print(f"Avg density of new training graphs: {avg_density}")
    # print(f"Median num edges of new training graphs: {median_num_edges}")
    # print(f"Median density of new training graphs: {median_density}")
    train_dataset = new_dataset[:new_train_nums]
    random.shuffle(train_dataset)
    val_dataset = new_dataset[new_train_nums:new_train_val_nums]
    test_dataset = new_dataset[new_train_val_nums:]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    if model_name == "GIN":
        model = GIN(num_features=num_features, num_classes=num_classes, num_hidden=num_hidden).to(
            device)
    elif model_name == "GCN":
        model = GCN(in_channels=num_features, hidden_channels=num_hidden, out_channels=num_classes,
                    num_layers=4).to(device)
    elif model_name == "TopKPool":
        model = TopKNet(in_channels=num_features, hidden_channels=num_hidden,
                        out_channels=num_classes).to(device)
    elif model_name == "DiffPool":
        model = DiffPoolNet(in_channels=num_features, hidden_channels=num_hidden,
                            out_channels=num_classes, max_nodes=median_num_nodes).to(device)
    elif model_name == "MinCutPool":
        model = MinCutPoolNet(in_channels=num_features, hidden_channels=num_hidden,
                              out_channels=num_classes, max_nodes=median_num_nodes).to(device)
    else:
        model = None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    max_val_acc = 0
    model_test_acc = 0
    model_test_loss = 0
    model_val_loss = 0
    best_epoch = 0
    train_losses = []
    val_losses = []
    test_losses = []
    for epoch in range(1, epochs):
        if aug == 'DropEdge':
            new_train_loader = augment_dataset_dropedge(train_loader, 0.2)
        elif aug == 'DropNode':
            new_train_loader = augment_dataset_dropnode(train_loader, 0.2)
        elif aug == 'Subgraph':
            new_train_loader = augment_dataset_subgraph(train_loader, 0.1)
        else:
            new_train_loader = train_loader
        model, train_loss, train_acc = train(model, new_train_loader, num_classes, optimizer)
        val_acc, val_loss = test(model, val_loader, num_classes)
        test_acc, test_loss = test(model, test_loader, num_classes)
        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        print(f'Finished epoch {epoch}')
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            model_test_loss = test_loss
            model_test_acc = test_acc
            model_val_loss = val_loss
            best_epoch = epoch
        # if epoch%20==0:
        #    print(
        #        'Epoch: {:03d}, Train Loss: {:.6f}, Val Loss: {:.6f}, Test Loss: {:.6f}, Train acc: {: .6f}, Val Acc: {: .6f}, Test Acc: {: .6f}'.format(
        #            epoch, train_loss, val_loss, test_loss, train_acc, val_acc, test_acc))

    end = time.time()
    total_time = f'{end - start:.2f}'
    with open('../results/train_log_exp3.csv', 'a') as f:
        f.write(
            f'{dataset_name}, {model_name}, {seed}, {aug}, {best_epoch}, {model_test_acc:.6f}, {model_test_loss:.4f}, {max_val_acc:.6f}, {model_val_loss:.4f}, {device}, {total_time}\n')
    if model_name == 'GCN':
        with open('../results/losses.txt', 'a') as f:
            f.write(
                f'{dataset_name}, {seed}, train, {train_losses}\n{dataset_name}, {seed}, val, {val_losses}\n{dataset_name}, {seed}, test, {test_losses}\n')
    print(
        f'Job ID: {id}, Dataset: {dataset_name}, Model: {model_name}, Seed: {seed}, Aug: {aug}, Best epoch: {best_epoch}, Test acc: {model_test_acc}, Test loss: {model_test_loss}, Val acc: {max_val_acc}, Val loss: {model_val_loss}')


if __name__ == '__main__':
    dataset_names = ['IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']
    models = ['GCN', 'GIN', 'MinCutPool', 'DiffPool', 'TopKPool']
    seeds = [1314, 11314, 21314, 31314, 41314, 51314, 61314, 71314, 0, 546464]
    augmentations = ['Vanilla', 'G-Mixup', 'Subgraph', 'DropEdge', 'DropNode']

    path = Path('../results/train_log_exp3.csv')
    if not path.is_file():
        with open(path, 'w') as f:
            f.write('Dataset, Model, Seed, Aug, BestEpoch, TestAcc, TestLoss, ValAcc, ValLoss, Device, Time\n')

    combination_list = []
    for dataset_name in dataset_names:
        for model in models:
            for seed in seeds:
                for aug in augmentations:
                    combination_list.append({'dataset_name': dataset_name, 'model': model, 'seed': seed, 'aug': aug})

    print(f'Possible combinations: {len(combination_list)}')

    for i, comb in enumerate(combination_list):
        if i >= 522:
            run_test(i, comb['dataset_name'], comb['model'], comb['seed'], comb['aug'])

