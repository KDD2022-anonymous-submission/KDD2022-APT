import io
import itertools
import os
import os.path as osp
from collections import defaultdict, namedtuple
import tqdm
import dgl
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
import scipy.sparse as sparse
import sklearn.preprocessing as preprocessing
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from dgl.data.tu import TUDataset
from scipy.sparse import linalg
from loader_gcc import PretrainDataset
from tkinter import _flatten
import networkx as nx
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.utils import remove_isolated_nodes, contains_isolated_nodes
from copy import deepcopy
import time
import math
from component.load_data import *


def get_hop2(node, G):
    seen = set()
    for n in list(G[node]):
        seen |= set(G[n])
    seen |= set([node])
    return len(seen)


def dgl_to_pyg(dgl_graph):
    temp = list(dgl_graph.all_edges())
    # temp = torch.tensor(temp, dtype=torch.long)
    edge = torch.stack((temp[0], temp[1]))
    print("exchange finished")
    return Data(edge_index=edge, x=None)


def rewire_graph(dgl_graph, rate):
    pyg_graph = dgl_to_pyg(dgl_graph)
    g = to_networkx(pyg_graph)
    degree_list = []
    degrees = g.degree()
    for key, value in degrees:
        degree_list.append(value)
    # degree_list_real, g = load_dataset("wiki")
    num_edge = len(g.edges)
    num_node = len(g.nodes)
    sam_num = int(num_edge * rate)
    idx_list = [i for i in range(num_edge)]
    degree_list = []
    degrees = g.degree()
    for key, value in degrees:
        degree_list.append(value)
    for i in range(sam_num):
        edge_list = list(g.edges())
        sam_list = random.sample(idx_list, 1)
        (a1, b1) = edge_list[sam_list[0]]
        count = 0
        while True:
            count = count + 1
            node_a = random.randint(0, num_node - 1)
            if count > 100000:
                break
            if degree_list[node_a] < degree_list[b1] or node_a == a1 or g.has_edge(a1, node_a) is True:
                continue
            else:
                degree_list[b1] = degree_list[b1] - 1
                degree_list[node_a] = degree_list[node_a] + 1
                g.remove_edges_from([(a1, b1)])
                g.add_edges_from([(a1, node_a)])
                break
    sum = 0
    num_edge = len(g.edges)
    for j in degree_list:
        if j != 0:
            sum = sum + j * math.log(j)
    entropy = sum / (2 * num_edge)
    print("entropy: ", entropy)
    dgl_final = dgl.DGLGraph(g)
    return dgl_final

def rewire_graph_max(dgl_graph, rate):
    degree_list_real, g = load_dataset("wiki")
    num_edge = len(g.edges)
    num_node = len(g.nodes)
    sam_num = int(num_edge * rate)
    idx_list = [i for i in range(num_edge)]
    degree_list = []
    degrees = g.degree()
    for key, value in degrees:
        degree_list.append(value)
    for i in range(sam_num):
        edge_list = list(g.edges())
        sam_list = random.sample(idx_list, 1)
        (a1, b1) = edge_list[sam_list[0]]
        count = 0
        zipped = zip(degree_list, idx_list)
        sort_zipped = sorted(zipped, key=lambda x: (x[0], x[1]))
        result = zip(*sort_zipped)
        sort_degrees, sort_idxs = [list(x) for x in result]
        # for j in range(len(sort_degrees)):
        #     if sort_degrees[j] < degree_list[b1]:
        #         j = j + 1
        #     else:
        #         break
        count = 0
        while True:
            s = random.randint(num_node - 10, num_node - 1)
            count = count + 1
            if count > 10000:
                break
            node_a = sort_idxs[s]
            if node_a == a1 or g.has_edge(a1, node_a) is True:
                continue
            else:
                degree_list[b1] = degree_list[b1] - 1
                degree_list[node_a] = degree_list[node_a] + 1
                g.remove_edges_from([(a1, b1)])
                g.add_edges_from([(a1, node_a)])
                break
    sum = 0
    num_edge = len(g.edges)
    for j in degree_list:
        if j != 0:
            sum = sum + j * math.log(j)
    entropy = sum / (2 * num_edge)
    print("entropy: ", entropy)
    dgl_final = dgl.DGLGraph(g)
    return dgl_final

def rewire_total_max(graph_list, rate):
    print("max")
    gg = []
    len1 = []
    for i in range(0, len(graph_list)):
        print("Start to flip the graph", i)
        t1 = time.time()
        temp = rewire_graph_max(graph_list[i], rate)
        t2 = time.time()
        print("graph finished", t2 - t1)
        gg.append(temp)
        len1.append(len(temp.nodes()))
    graph_len = torch.LongTensor(len1)
    graph_dic = {}
    graph_dic['graph_sizes'] = graph_len
    dgl.data.utils.save_graphs("/home/syf/gcc_modified/rewire_" + str(rate) + ".bin", gg, graph_dic)

def rewire_total(graph_list, rate):
    print("origin")
    gg = []
    len1 = []
    for i in range(0, len(graph_list)):
        print("Start to flip the graph", i)
        t1 = time.time()
        temp = rewire_graph(graph_list[i], rate)
        t2 = time.time()
        print("graph finished", t2 - t1)
        gg.append(temp)
        len1.append(len(temp.nodes()))
    graph_len = torch.LongTensor(len1)
    graph_dic = {}
    graph_dic['graph_sizes'] = graph_len
    dgl.data.utils.save_graphs("/home/syf/gcc_modified/rewire_" + str(rate) + ".bin", gg, graph_dic)


def rewire_total1(graph_list, rate):
    gg = []
    len1 = []
    for i in range(0, len(graph_list)):
        print("Start to flip the graph", i)
        t1 = time.time()
        temp = rewire_graph(graph_list[i], rate)
        t2 = time.time()
        print("graph finished", t2 - t1)
        gg.append(temp)
        len1.append(len(temp.nodes()))
    graph_len = torch.LongTensor(len1)
    graph_dic = {}
    graph_dic['graph_sizes'] = graph_len
    dgl.data.utils.save_graphs("/home/syf/gcc_modified/rewire_" + str(rate) + ".bin", gg, graph_dic)


def flip_edges(dgl_graph, flip_rate):
    pyg_graph = dgl_to_pyg(dgl_graph)
    edge_index = pyg_graph.edge_index
    print("The shape of edges before noise", edge_index.shape)
    edge_num = edge_index.shape[-1]  # adj = to_dense_adj(edge_index)
    print(edge_num)
    nds_in_egs = list(set(_flatten(edge_index.tolist())))
    node_num = len(nds_in_egs)
    possible_edge_num = node_num * (node_num - 1)

    draw_size = edge_num * flip_rate  # m
    draw_res = np.random.choice(a=possible_edge_num - 1, size=int(draw_size), replace=False)

    # flip edges
    nx_graph = to_networkx(pyg_graph)
    edge = deepcopy(list(nx_graph.edges()))
    edge1 = edge_index.t()
    edge = [torch.tensor(i) for i in edge]
    remove_es, add_es = [], []
    i = 0
    print("phase1")
    for it in draw_res:
        i = int(it / (node_num - 1))
        j = int((it % (node_num - 1)) if (it % (node_num - 1)) < i else ((it % (node_num - 1)) + 1))
        assert i < node_num and j < node_num, "Fuck you!" + str(it) + '_' + str(i) + '_' + str(j) + '_' + str(
            node_num)
        chosen_edge = torch.tensor([nds_in_egs[i], nds_in_egs[j]])
        # if chosen_edge in edge_index.t():  # del the 1
        if chosen_edge in edge_index.t():
            idx = (edge_index.t() == chosen_edge).nonzero(as_tuple=True)[0][0]
            if idx not in remove_es and int(idx) < edge_num:
                remove_es.append(idx)
        else:
            if idx not in chosen_edge:  # insert the 1
                add_es.append(chosen_edge)
        i += 1
        print(i)
    print("phase2")
    for idx in remove_es:
        u, v = edge[idx]
        print(u, v)
        nx_graph.remove_edge(int(u), int(v))
    for e in add_es:
        print(u, v)
        u, v = e
        nx_graph.add_edge(int(u), int(v))
    pyg_graph = from_networkx(nx_graph)
    print("The shape of edges after noise", pyg_graph.edge_index.shape)
    g = dgl.DGLGraph()
    g.from_networkx(nx_graph)
    return g
    # check nodes
    new_nodes = _flatten(pyg_graph.edge_index.tolist())
    assert set(nds_in_egs) >= set(new_nodes), "Fucking wrong with nodes"
    if pyg_graph.contains_isolated_nodes():
        pyg_graph.edge_index, _, node_mask = remove_isolated_nodes(pyg_graph.edge_index)
    return pyg_graph


def filp_total(graph_list, rate):
    gg = []
    len1 = []
    for i in range(0, len(graph_list)):
        print("Start to flip the graph", i)
        t1 = time.time()
        temp = flip_edges1(graph_list[i], rate)
        t2 = time.time()
        print("graph finished", t2 - t1)
        gg.append(temp)
        len1.append(len(temp.nodes()))
    graph_len = torch.LongTensor(len1)
    graph_dic = {}
    graph_dic['graph_sizes'] = graph_len
    dgl.data.utils.save_graphs("./data/pre_small.bin", gg, graph_dic)


def flip_edges1(dgl_graph, rate):
    pyg_graph = dgl_to_pyg(dgl_graph)
    edge_index = pyg_graph.edge_index

    def baseline_random_top_flips(candidates, n_flips):
        a = np.random.permutation(candidates.shape[0])
        return candidates[a[:n_flips]]

    num_node = edge_index.shape[1]
    n_candidates = 10000
    num_flip = int(dgl_graph.batch_num_edges[0] * rate)
    print("num_flip", num_flip)
    candidates = np.random.randint(0, dgl_graph.batch_num_nodes[0], [n_candidates * 5, 2])
    candidates = candidates[candidates[:, 0] < candidates[:, 1]]
    # candidates = candidates[_A_obs[candidates[:, 0], candidates[:, 1]].A1 == 0]
    candidates = np.array(list(set(map(tuple, candidates))))
    candidates_all = candidates[:n_candidates]
    print("# of candidates:", len(candidates_all))
    our_restart_flips = baseline_random_top_flips(candidates_all, num_flip)

    t1 = time.time()
    nx_graph = to_networkx(pyg_graph)
    t2 = time.time()
    print("to network1", t2 - t1)

    t1 = time.time()
    for i in range(our_restart_flips.shape[0]):
        chosen_edge = torch.tensor([our_restart_flips[i][0], our_restart_flips[i][1]])
        k = nx_graph.edges()
        if chosen_edge in k:
            nx_graph.remove_edge(our_restart_flips[i][0], our_restart_flips[i][1])
        else:
            nx_graph.add_edge(our_restart_flips[i][0], our_restart_flips[i][1])
    t2 = time.time()
    print("transform", t2 - t1)
    t1 = time.time()
    g = dgl.DGLGraph()
    g.from_networkx(nx_graph)
    t2 = time.time()
    print("to dgl", t2 - t1)
    return g


def flip_dglgraph(dgl_graph):
    pass


def dgl_add_edge():
    pass
