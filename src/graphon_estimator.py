# some code are from https://github.com/HongtengXu/SGWB-Graphon

import copy
# import cv2
import numpy as np
import torch

from skimage.restoration import denoise_tv_chambolle
from typing import List

from torch import Tensor


def graph_numpy2tensor(graphs: List[Tensor]) -> torch.Tensor:
    """
    Convert a list of np arrays to a pytorch tensor
    :param graphs: [K (N, N) adjacency matrices]
    :return:
        graph_tensor: [K, N, N] tensor
    """
    graph_tensor = np.array(graphs)
    return torch.from_numpy(graph_tensor).float()


def estimate_target_distribution(probs: List[np.ndarray], dim_t: int = None) -> np.ndarray:
    """
    Estimate target distribution via the average of sorted source probabilities
    Args:
        probs: a list of node distributions [(n_s, 1) the distribution of source nodes]
        dim_t: the dimension of target distribution
    Returns:
        p_t: (dim_t, 1) vector representing a distribution
    """
    if dim_t is None:
        dim_t = min([probs[i].shape[0] for i in range(len(probs))])

    p_t = np.zeros((dim_t, 1))
    x_t = np.linspace(0, 1, p_t.shape[0])
    for i in range(len(probs)):
        p_s = probs[i][:, 0]
        p_s = np.sort(p_s)[::-1]
        x_s = np.linspace(0, 1, p_s.shape[0])
        p_t_i = np.interp(x_t, x_s, p_s) + 1e-3
        p_t[:, 0] += p_t_i

    p_t /= np.sum(p_t)
    return p_t


# def estimate_graphon(graphs: List[np.ndarray], method, args) -> Tuple[np.ndarray, np.ndarray]:
#     if method == 'GWB' or method == 'SGWB' or method == 'FGWB' or method == 'SFGWB':
#         aligned_graphs, normalized_node_degrees, max_num, min_num = align_graphs(graphs, padding=False)
#     else:
#         aligned_graphs, normalized_node_degrees, max_num, min_num = align_graphs(graphs, padding=True)
#
#     block_size = int(np.log2(max_num) + 1)
#     num_blocks = int(max_num / block_size)
#     p_b = estimate_target_distribution(normalized_node_degrees, dim_t=num_blocks)
#
#
#     stepfunc = universal_svd(aligned_graphs, threshold=args.threshold_usvt)
#
#     graphon = cv2.resize(stepfunc, dsize=(args.r, args.r), interpolation=cv2.INTER_LINEAR)
#     return stepfunc, graphon



# In the paper they said that they used LG as the step function approximator, check with authors !!
def universal_svd(aligned_graphs: List[Tensor], threshold: float = 2.02, sum_graph: Tensor = None) -> np.ndarray:
    """
    Estimate a graphon by universal singular value thresholding.
    Reference:
    Chatterjee, Sourav.
    "Matrix estimation by universal singular value thresholding."
    The Annals of Statistics 43.1 (2015): 177-214.
    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param threshold: the threshold for singular values
    :return: graphon: the estimated (r, r) graphon model
    """

    if sum_graph is None:
        aligned_graphs = graph_numpy2tensor(aligned_graphs)
        num_graphs = aligned_graphs.size(0)

        if num_graphs > 1:
            sum_graph = torch.mean(aligned_graphs, dim=0)
        else:
            sum_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = sum_graph.size(0)
    print('Doing SVD of matrix of size: ', num_nodes)
    u, s, v = torch.svd(sum_graph)
    print('Finished SVD!')
    singular_threshold = threshold * (num_nodes ** 0.5)
    binary_s = torch.lt(s, singular_threshold)
    s[binary_s] = 0
    graphon = u @ torch.diag(s) @ torch.t(v)
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    graphon = graphon.numpy()
    return graphon


def guess_rank(matrix: torch.Tensor) -> int:
    """
    A function to guess the rank of a matrix
    :param matrix: a torch.Tensor matrix
    :return:
    """
    n = matrix.size(0)
    m = matrix.size(1)
    epsilon = torch.sum(matrix != 0) / ((n * m) ** 0.5)

    u, s, v = torch.svd(matrix, compute_uv=False)
    max_num = min([100, s.size(0)])
    s = s[:max_num]
    s, _ = torch.sort(s, descending=True)
    diff_s1 = s[:-1] - s[1:]
    diff_s1 = diff_s1 / torch.mean(diff_s1[-10:])
    r1 = torch.zeros(1)
    gamma = 0.05
    while r1.item() <= 0:
        cost = torch.zeros(diff_s1.size(0))
        for i in range(diff_s1.size(0)):
            cost[i] = gamma * torch.max(diff_s1[i:]) + i + 1

        idx = torch.argmin(cost)
        r1 = torch.argmax(idx)
        gamma += 0.05

    cost = torch.zeros(diff_s1.size(0))
    for i in range(diff_s1.size(0)):
        cost[i] = s[i + 1] + ((i + 1) * epsilon ** 0.5) * s[0] / epsilon

    idx = torch.argmin(cost)
    r2 = torch.max(idx)
    return max([r1.item(), r2.item()])



def averaging_graphs(aligned_graphs: List[np.ndarray], trans: List[np.ndarray], ws: np.ndarray) -> np.ndarray:
    """
    sum_k w_k * (Tk @ Gk @ Tk')
    :param aligned_graphs: a list of (Ni, Ni) adjacency matrices
    :param trans: a list of (Nb, Ni) transport matrices
    :param ws: (K, ) weights
    :return: averaged_graph: a (Nb, Nb) adjacency matrix
    """
    averaged_graph = 0
    for k in range(ws.shape[0]):
        averaged_graph += ws[k] * (trans[k] @ aligned_graphs[k] @ trans[k].T)
    return averaged_graph


def proximal_ot(cost: np.ndarray,
                p1: np.ndarray,
                p2: np.ndarray,
                iters: int,
                beta: float,
                error_bound: float = 1e-10,
                prior: np.ndarray = None) -> np.ndarray:
    """
    min_{T in Pi(p1, p2)} <cost, T> + beta * KL(T | prior)
    :param cost: (n1, n2) cost matrix
    :param p1: (n1, 1) source distribution
    :param p2: (n2, 1) target distribution
    :param iters: the number of Sinkhorn iterations
    :param beta: the weight of proximal term
    :param error_bound: the relative error bound
    :param prior: the prior of optimal transport matrix T, if it is None, the proximal term degrades to Entropy term
    :return:
        trans: a (n1, n2) optimal transport matrix
    """
    if prior is not None:
        kernel = np.exp(-cost / beta) * prior
    else:
        kernel = np.exp(-cost / beta)

    relative_error = np.inf
    a = np.ones(p1.shape) / p1.shape[0]
    b = []
    i = 0

    while relative_error > error_bound and i < iters:
        b = p2 / (np.matmul(kernel.T, a))
        a_new = p1 / np.matmul(kernel, b)
        relative_error = np.sum(np.abs(a_new - a)) / np.sum(np.abs(a))
        a = copy.deepcopy(a_new)
        i += 1
    trans = np.matmul(a, b.T) * kernel
    return trans


def node_cost_st(cost_s: np.ndarray, cost_t: np.ndarray, p_s: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    """
    Calculate invariant cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        p_s: (n_s, 1) array, the distribution of source nodes
        p_t: (n_t, 1) array, the distribution of target nodes
    Returns:
        cost_st: (n_s, n_t) array, the estimated invariant cost between the nodes in two graphs
    """
    n_s = cost_s.shape[0]
    n_t = cost_t.shape[0]
    f1_st = np.repeat((cost_s ** 2) @ p_s, n_t, axis=1)
    f2_st = np.repeat(((cost_t ** 2) @ p_t).T, n_s, axis=0)
    cost_st = f1_st + f2_st
    return cost_st


def gw_cost(cost_s: np.ndarray, cost_t: np.ndarray, trans: np.ndarray, p_s: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    """
    Calculate the cost between the nodes in different graphs based on learned optimal transport
    Args:
        cost_s: (n_s, n_s) array, the cost matrix of source graph
        cost_t: (n_t, n_t) array, the cost matrix of target graph
        trans: (n_s, n_t) array, the learned optimal transport between two graphs
        p_s: (n_s, 1) array, the distribution of source nodes
        p_t: (n_t, 1) array, the distribution of target nodes
    Returns:
        cost: (n_s, n_t) array, the estimated cost between the nodes in two graphs
    """
    cost_st = node_cost_st(cost_s, cost_t, p_s, p_t)
    return cost_st - 2 * (cost_s @ trans @ cost_t.T)

def largest_gap(aligned_graphs: List[Tensor], k: int, sum_graph: Tensor = None) -> np.ndarray:
    """
    Estimate a graphon by a stochastic block model based n empirical degrees
    Reference:
    Channarond, Antoine, Jean-Jacques Daudin, and Stéphane Robin.
    "Classification and estimation in the Stochastic Blockmodel based on the empirical degrees."
    Electronic Journal of Statistics 6 (2012): 2574-2601.
    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param k: the number of blocks
    :return: a (r, r) estimation of graphon
    """
    if sum_graph is None:
        aligned_graphs = graph_numpy2tensor(aligned_graphs)
        num_graphs = aligned_graphs.size(0)

        if num_graphs > 1:
            sum_graph = torch.mean(aligned_graphs, dim=0)
        else:
            sum_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = sum_graph.size(0)

    # sort node degrees
    degree = torch.sum(sum_graph, dim=1)
    sorted_degree = degree / (num_nodes - 1)
    idx = torch.arange(0, num_nodes)

    # find num_blocks-1 largest gap of the node degrees
    diff_degree = sorted_degree[1:] - sorted_degree[:-1]
    _, index = torch.topk(diff_degree, k=k - 1)
    sorted_index, _ = torch.sort(index + 1, descending=False)
    blocks = {}
    for b in range(k):
        if b == 0:
            blocks[b] = idx[0:sorted_index[b]]
        elif b == k - 1:
            blocks[b] = idx[sorted_index[b - 1]:num_nodes]
        else:
            blocks[b] = idx[sorted_index[b - 1]:sorted_index[b]]

    # derive the graphon by stochastic block model
    probability = torch.zeros(k, k)
    graphon = torch.zeros(num_nodes, num_nodes)
    for i in range(k):
        for j in range(k):
            rows = blocks[i]
            cols = blocks[j]
            tmp = sum_graph[rows, :]
            tmp = tmp[:, cols]
            probability[i, j] = torch.sum(tmp) / (rows.size(0) * cols.size(0))
            for r in range(rows.size(0)):
                for c in range(cols.size(0)):
                    graphon[rows[r], cols[c]] = probability[i, j]
    graphon = graphon.numpy()
    return graphon

def matrix_completion(aligned_graphs: List[np.ndarray], rank: int = None) -> np.ndarray:
    """
    Estimate the graphon by matrix completion
    Reference:
    Keshavan, Raghunandan H., Andrea Montanari, and Sewoong Oh.
    "Matrix completion from a few entries."
    IEEE transactions on information theory 56.6 (2010): 2980-2998.
    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param rank: the rank of adjacency matrix
    :return: graphon: the estimated graphon model
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs)
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        average_graph = torch.mean(aligned_graphs, dim=0)
    else:
        average_graph = aligned_graphs[0, :, :]  # (N, N)

    num_nodes = average_graph.size(0)

    # low-rank approximation via svd
    average_graph = 2 * (average_graph - 0.5)
    if rank is None:
        rank = int(num_nodes / int(np.log(num_nodes)))
        # rank = guess_rank(average_graph)
        # print(rank)
    u, s, v = torch.svd(average_graph)
    graphon = (u[:, :rank] @ torch.diag(s[:rank]) @ torch.t(v[:, :rank]) + 1) / 2
    graphon[graphon > 1] = 1
    graphon[graphon < 0] = 0
    graphon = graphon.numpy()
    return graphon


def sorted_smooth(aligned_graphs: List[np.ndarray], h: int) -> np.ndarray:
    """
    Estimate a graphon by a sorting and smoothing method
    Reference:
    S. H. Chan and E. M. Airoldi,
    "A Consistent Histogram Estimator for Exchangeable Graph Models",
    Proceedings of International Conference on Machine Learning, 2014.
    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param h: the block size
    :return: a (k, k) step function and  a (r, r) estimation of graphon
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs)
    num_graphs = aligned_graphs.size(0)

    if num_graphs > 1:
        sum_graph = torch.mean(aligned_graphs, dim=0, keepdim=True).unsqueeze(0)
    else:
        sum_graph = aligned_graphs.unsqueeze(0)  # (1, 1, N, N)

    # histogram of graph
    kernel = torch.ones(1, 1, h, h) / (h ** 2)
    # print(sum_graph.size(), kernel.size())
    graphon = torch.nn.functional.conv2d(sum_graph, kernel, padding=0, stride=h, bias=None)
    graphon = graphon[0, 0, :, :].numpy()
    # total variation denoising
    graphon = denoise_tv_chambolle(graphon, weight=h)
    return graphon

def estimate_blocks_directed(aligned_graphs: List[np.ndarray], threshold: float) -> np.ndarray:
    """
    Estimate a graphon by stochastic block approximation.
    Reference:
    E. M. Airoldi, T. B. Costa, and S. H. Chan,
    "Stochastic blockmodel approximation of a graphon: Theory and consistent estimation",
    Advances in Neural Information Processing Systems 2013.
    :param aligned_graphs: a list of (N, N) adjacency matrices
    :param threshold: the threshold for singular values
    :return: graphon: the estimated (r, r) graphon model
    """
    aligned_graphs = graph_numpy2tensor(aligned_graphs)
    num_graphs = aligned_graphs.size(0)
    num_nodes = aligned_graphs.size(1)
    aligned_graphs = aligned_graphs.permute(1, 2, 0)

    num_half_graphs = int(num_graphs / 2)
    w = 1/((num_half_graphs + 1) * (num_nodes - num_half_graphs))
    if num_graphs > 1:
        sum_graph = torch.sum(aligned_graphs, dim=2)
        if num_half_graphs > 1:
            sum_half_graph1 = torch.sum(aligned_graphs[:, :, :num_half_graphs], dim=2)
        else:
            sum_half_graph1 = aligned_graphs[:, :, 0]

        if num_graphs - num_half_graphs > 1:
            sum_half_graph2 = torch.sum(aligned_graphs[:, :, num_half_graphs:], dim=2)
        else:
            sum_half_graph2 = aligned_graphs[:, :, -1]
    else:
        sum_half_graph1 = aligned_graphs[:, :, 0]
        sum_half_graph2 = aligned_graphs[:, :, 0]
        sum_graph = aligned_graphs[:, :, 0]

    pivot_idx = [0]
    blocks = dict()
    blocks[0] = [pivot_idx[0]]
    not_assigned_idx = list(range(num_nodes))
    not_assigned_idx.remove(pivot_idx[0])
    while len(not_assigned_idx) > 0:
        if len(not_assigned_idx) == 1:
            i = not_assigned_idx[0]
        else:
            idx = np.random.permutation(len(not_assigned_idx))
            i = not_assigned_idx[idx[0]]
        not_assigned_idx.remove(i)

        dhat = torch.zeros(len(pivot_idx))
        for j in range(len(pivot_idx)):
            bj = pivot_idx[j]

            set_idx = list(range(num_nodes))
            set_idx.remove(i)
            set_idx.remove(bj)

            term1 = torch.sum(w * sum_half_graph1[i, set_idx] * sum_half_graph2[i, set_idx])
            term2 = torch.sum(w * sum_half_graph1[bj, set_idx] * sum_half_graph2[bj, set_idx])
            term3 = torch.sum(w * sum_half_graph1[i, set_idx] * sum_half_graph2[bj, set_idx])
            term4 = torch.sum(w * sum_half_graph1[bj, set_idx] * sum_half_graph2[i, set_idx])

            term5 = torch.sum(w * sum_half_graph1[set_idx, i] * sum_half_graph2[set_idx, i])
            term6 = torch.sum(w * sum_half_graph1[set_idx, bj] * sum_half_graph2[set_idx, bj])
            term7 = torch.sum(w * sum_half_graph1[set_idx, i] * sum_half_graph2[set_idx, bj])
            term8 = torch.sum(w * sum_half_graph1[set_idx, bj] * sum_half_graph2[set_idx, i])

            dhat[j] = (0.5*(torch.abs(term1 + term2 - term3 - term4) +
                            torch.abs(term5 + term6 - term7 - term8)) / len(set_idx)) ** 0.5

        if dhat.size(0) == 1:
            value = dhat[0]
            idx = 0
        else:
            value, idx = torch.min(dhat, dim=0)
            idx = idx.item()

        if value < threshold:
            blocks[idx].append(i)
        else:
            blocks[len(pivot_idx)] = [i]
            pivot_idx.append(i)

    num_blocks = len(blocks)
    for key in blocks.keys():
        tmp = np.array(blocks[key])
        blocks[key] = torch.from_numpy(tmp).long()
    print(num_blocks)

    # # derive the graphon by stochastic block model
    probability = torch.zeros(num_blocks, num_blocks)
    graphon = torch.zeros(num_nodes, num_nodes)
    for i in range(num_blocks):
        for j in range(num_blocks):
            rows = blocks[i]
            cols = blocks[j]
            tmp = sum_graph[rows, :]
            tmp = tmp[:, cols]
            probability[i, j] = torch.sum(tmp) / (num_graphs * rows.size(0) * cols.size(0))
            for r in range(rows.size(0)):
                for c in range(cols.size(0)):
                    graphon[rows[r], cols[c]] = probability[i, j]

    graphon = graphon.numpy()
    return graphon