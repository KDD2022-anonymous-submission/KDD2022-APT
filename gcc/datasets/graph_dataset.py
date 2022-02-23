#!/usr/bin/env python
# encoding: utf-8
# File Name: graph_dataset.py
# Author: Jiezhong Qiu
# Create Time: 2019/12/11 12:17
# TODO:

import math
import operator
import dgl
import dgl.data
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from dgl.data import AmazonCoBuy, Coauthor
# from dgl.nodeflow import NodeFlow
import time
from gcc.datasets import data_util
from gcc.datasets.data_util import batcher
from gcc.contrastive.criterions import NCESoftmaxLoss


def z_score_normalization(sequence):
    return (np.array(sequence) - np.mean(sequence)) / np.std(sequence)


def compute_alpha(degree_sequence, d_min):
    """
    Approximate the alpha of a power law distribution.
    Parameters
    ----------
    degree_sequence: degree sequence
    d_min: int
        The minimum degree of nodes to consider
    Returns
    -------
    alpha: float
        The estimated alpha of the power law distribution
    """
    S_d = np.sum(np.log(degree_sequence[degree_sequence >= d_min]))
    n = np.sum(degree_sequence >= d_min)
    return n / (S_d - n * np.log(d_min - 0.5)) + 1


def get_loss(model, graph_idx, dgl_file, contrast, device, loss_kind):
    train_dataset = LoadBalanceGraphDataset_al(
        rw_hops=256,
        restart_prob=0.8,
        positional_embedding_size=32,
        num_workers=1,
        num_copies=1,
        num_samples=500,
        dgl_graphs_file=dgl_file,
        single_graph=True,
        sample_graph_idx=graph_idx
    )

    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_dataset,
    #     batch_size=32,
    #     collate_fn=batcher(),
    #     shuffle=False,
    #     num_workers=1,
    # )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=32,
        collate_fn=batcher(),
        shuffle=False,
        num_workers=1,
        worker_init_fn=worker_init_fn,
    )
    model.eval()

    loss_list = []
    criterion = NCESoftmaxLoss()
    for idx, batch in enumerate(train_loader):
        graph_q, graph_k = batch
        bsz = graph_q.batch_size
        graph_q.to(device)
        graph_k.to(device)
        with torch.no_grad():
            feat_q = model(graph_q)
            feat_k = model(graph_k)
        if loss_kind == "nceloss":
            out = contrast(feat_q, feat_k)
            ori_loss = criterion(out)
            loss_list.append(ori_loss.item())
        elif loss_kind == "max_uncertainty":
            l_pos, max_neg = contrast(feat_q, feat_k, update=False, ret_max=True)
            uncertain = -np.log(np.array(l_pos) / np.array(max_neg)).mean()
            loss_list.append(uncertain)
    return np.mean(loss_list)


def select_graph(dgl_graphs, candidate_indices, model, dgl_file, contrast, epoch, device):
    if len(candidate_indices) == 0:
        return -1
    degree_mean_list = []
    entropy_list = []
    std_list = []
    density_list = []
    alpha_list = []
    x_min = 1
    # graphs = sorted(enumerate(dgl_graphs[1]["graph_sizes"]), key=operator.itemgetter(1), reverse=True)
    graphs = [dgl_graphs[0][i] for i in candidate_indices]
    loss_score_list = []
    g_para = 1
    e_para = 5
    loss_type = "nceloss"
    time_type = "beta"
    print("time type:", time_type)
    print("loss type:", loss_type)
    if time_type == "beta":
        g_para = np.random.beta(1, 3 - 0.995 ** epoch)
        e_para = 1 - g_para
    elif time_type == "cos":
        g_para = np.cos(epoch / 100 * math.pi / 2)
        e_para = 1 - g_para
    print("gamma:", e_para)
    for i, graph in enumerate(graphs):
        dglgraph = dgl.to_bidirected(graph)
        num_nodes, num_edges = dglgraph.batch_num_nodes[0], dglgraph.batch_num_edges[0] / 2
        degrees = dglgraph.in_degrees(range(num_nodes)).tolist()
        degree_mean_list.append(np.mean(degrees))
        entropy_list.append(float(sum([degrees[i] * math.log(degrees[i]) for i in range(num_nodes)]) / 2 / num_edges))
        std_list.append(math.pow(np.std(degrees), 2))
        density_list.append(2 * num_edges / num_nodes / (num_nodes - 1))
        alpha_list.append(compute_alpha(np.array(degrees), x_min))
        if model is not None:
            loss_score = get_loss(model, i, dgl_file, contrast, device, loss_kind=loss_type)
            loss_score_list.append(loss_score)
    entropy_list = z_score_normalization(entropy_list)
    std_list = z_score_normalization(std_list)
    density_list = z_score_normalization(density_list)
    degree_mean_list = z_score_normalization(degree_mean_list)
    # proerty_init_no_alpha = [list(item) for item in zip(entropy_list, std_list, density_list, degree_mean_list)]
    alpha_list = z_score_normalization(alpha_list)
    alpha_list_a = [-i for i in alpha_list]

    proerty_init_ = [list(item) for item in zip(entropy_list, std_list, density_list, degree_mean_list, alpha_list_a)]
    graph_based_score = [sum(proerty_init_[i]) / len(proerty_init_[i]) for i in
                         range(len(proerty_init_))]
    if model is not None:
        # embed_based_score = loss_score_list
        embed_based_score = z_score_normalization(loss_score_list)
        final_score = [graph_based_score[i] * g_para + embed_based_score[i] * e_para for i in range(len(graphs))]
    else:
        final_score = graph_based_score
    max_idx = np.argmax(final_score)
    max_idx = candidate_indices[max_idx]
    return max_idx


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset

    dataset.graphs, _ = dgl.data.utils.load_graphs(
        dataset.dgl_graphs_file, dataset.jobs[worker_id]
    )
    dataset.length = sum([g.number_of_nodes() for g in dataset.graphs])
    np.random.seed(worker_info.seed % (2 ** 32))


class LoadBalanceGraphDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            rw_hops=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
            num_workers=1,
            dgl_graphs_file="/data/srtpgroup/small.bin",
            num_samples=10000,
            num_copies=1,
            graph_transform=None,
            aug="rwr",
            num_neighbors=5,
    ):
        super(LoadBalanceGraphDataset).__init__()
        self.rw_hops = rw_hops
        self.num_neighbors = num_neighbors
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.num_samples = num_samples
        assert sum(step_dist) == 1.0
        assert positional_embedding_size > 1
        self.dgl_graphs_file = dgl_graphs_file
        graph_sizes = dgl.data.utils.load_labels(dgl_graphs_file)[
            "graph_sizes"
        ].tolist()
        print("load graph done")

        # a simple greedy algorithm for load balance
        # sorted graphs w.r.t its size in decreasing order
        # for each graph, assign it to the worker with least workload
        assert num_workers % num_copies == 0
        jobs = [list() for i in range(num_workers // num_copies)]
        workloads = [0] * (num_workers // num_copies)
        graph_sizes = sorted(
            enumerate(graph_sizes), key=operator.itemgetter(1), reverse=True
        )
        for idx, size in graph_sizes:
            argmin = workloads.index(min(workloads))
            workloads[argmin] += size
            jobs[argmin].append(idx)
        self.jobs = jobs * num_copies
        self.total = self.num_samples * num_workers
        self.graph_transform = graph_transform
        assert aug in ("rwr", "ns")
        self.aug = aug

    def __len__(self):
        return self.num_samples * num_workers

    def __iter__(self):
        degrees = torch.cat([g.in_degrees().double() ** 0.75 for g in self.graphs])
        prob = degrees / torch.sum(degrees)
        samples = np.random.choice(
            self.length, size=self.num_samples, replace=True, p=prob.numpy()
        )
        i = 0
        for idx in samples:
            i = i + 1
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.contrib.sampling.random_walk(
                g=self.graphs[graph_idx], seeds=[node_idx], num_traces=1, num_hops=step
            )[0][0][-1].item()

        if self.aug == "rwr":
            max_nodes_per_seed = max(
                self.rw_hops,
                int(
                    (
                            (self.graphs[graph_idx].in_degree(node_idx) ** 0.75)
                            * math.e
                            / (math.e - 1)
                            / self.restart_prob
                    )
                    + 0.5
                ),
            )
            traces = dgl.contrib.sampling.random_walk_with_restart(
                self.graphs[graph_idx],
                seeds=[node_idx, other_node_idx],
                restart_prob=self.restart_prob,
                max_nodes_per_seed=max_nodes_per_seed,
            )
        elif self.aug == "ns":
            prob = dgl.backend.tensor([], dgl.backend.float32)
            prob = dgl.backend.zerocopy_to_dgl_ndarray(prob)
            nf1 = dgl.contrib.sampling.sampler._CAPI_NeighborSampling(
                self.graphs[graph_idx]._graph,
                dgl.utils.toindex([node_idx]).todgltensor(),
                0,  # batch_start_id
                1,  # batch_size
                1,  # workers
                self.num_neighbors,  # expand_factor
                self.rw_hops,  # num_hops
                "out",
                False,
                prob,
            )[0]
            nf1 = NodeFlow(self.graphs[graph_idx], nf1)
            trace1 = [nf1.layer_parent_nid(i) for i in range(nf1.num_layers)]
            nf2 = dgl.contrib.sampling.sampler._CAPI_NeighborSampling(
                self.graphs[graph_idx]._graph,
                dgl.utils.toindex([other_node_idx]).todgltensor(),
                0,  # batch_start_id
                1,  # batch_size
                1,  # workers
                self.num_neighbors,  # expand_factor
                self.rw_hops,  # num_hops
                "out",
                False,
                prob,
            )[0]
            nf2 = NodeFlow(self.graphs[graph_idx], nf2)
            trace2 = [nf2.layer_parent_nid(i) for i in range(nf2.num_layers)]
            traces = [trace1, trace2]
        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        graph_k = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            positional_embedding_size=self.positional_embedding_size,
        )
        if self.graph_transform:
            graph_q = self.graph_transform(graph_q)
            graph_k = self.graph_transform(graph_k)
        return graph_q, graph_k


class LoadBalanceGraphDataset_al(torch.utils.data.IterableDataset):
    def __init__(
            self,
            rw_hops=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
            num_workers=1,
            dgl_graphs_file="/data/srtpgroup/small.bin",
            num_samples=10000,
            num_copies=1,
            graph_transform=None,
            aug="rwr",
            num_neighbors=5,
            single_graph=False,
            sample_graph_idx=0,
    ):
        super(LoadBalanceGraphDataset_al).__init__()
        self.rw_hops = rw_hops
        self.num_neighbors = num_neighbors
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.num_samples = num_samples
        assert sum(step_dist) == 1.0
        assert positional_embedding_size > 1
        self.dgl_graphs_file = dgl_graphs_file
        graph_sizes = dgl.data.utils.load_labels(dgl_graphs_file)[
            "graph_sizes"
        ].tolist()
        self.all_idx = [i for i in range(len(graph_sizes))]
        print("load graph done")

        assert num_workers % num_copies == 0
        jobs = [list() for i in range(num_workers // num_copies)]
        # workloads = [0] * (num_workers // num_copies)
        graph_sizes = sorted(
            enumerate(graph_sizes), key=operator.itemgetter(1), reverse=True
        )
        # for idx, size in graph_sizes:
        #     argmin = workloads.index(min(workloads))
        #     workloads[argmin] += size
        #     jobs[argmin].append(idx)
        self.device = None
        self.jobs = jobs * num_copies
        if single_graph == True:
            self.jobs[0] = [sample_graph_idx]
        self.total = self.num_samples * num_workers
        self.graph_transform = graph_transform
        assert aug in ("rwr", "ns")
        self.aug = aug

    def update(self, model, contrast, epoch, random_flag=False):
        print("Current dataset:", self.jobs)
        a = dgl.data.utils.load_graphs(self.dgl_graphs_file)
        idx = select_graph(a, list(set(self.all_idx) - set(self.jobs[0])), model, self.dgl_graphs_file, contrast, epoch,
                           self.device)
        if random_flag is True:
            if len(list(set(self.all_idx) - set(self.jobs[0]))) > 0:
                print("random")
                idx = int(np.random.choice(list(set(self.all_idx) - set(self.jobs[0])), 1)[0])
            else:
                idx = -1
        if idx != -1:
            self.jobs[0].append(idx)

    def __len__(self):
        return self.num_samples * num_workers

    def __iter__(self):
        degrees = torch.cat([g.in_degrees().double() ** 0.75 for g in self.graphs])
        prob = degrees / torch.sum(degrees)
        samples = np.random.choice(
            self.length, size=self.num_samples, replace=True, p=prob.numpy()
        )
        i = 0
        for idx in samples:
            i = i + 1
            yield self.__getitem__(idx)

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.contrib.sampling.random_walk(
                g=self.graphs[graph_idx], seeds=[node_idx], num_traces=1, num_hops=step
            )[0][0][-1].item()

        if self.aug == "rwr":
            max_nodes_per_seed = max(
                self.rw_hops,
                int(
                    (
                            (self.graphs[graph_idx].in_degree(node_idx) ** 0.75)
                            * math.e
                            / (math.e - 1)
                            / self.restart_prob
                    )
                    + 0.5
                ),
            )
            traces = dgl.contrib.sampling.random_walk_with_restart(
                self.graphs[graph_idx],
                seeds=[node_idx, other_node_idx],
                restart_prob=self.restart_prob,
                max_nodes_per_seed=max_nodes_per_seed,
            )
        elif self.aug == "ns":
            prob = dgl.backend.tensor([], dgl.backend.float32)
            prob = dgl.backend.zerocopy_to_dgl_ndarray(prob)
            nf1 = dgl.contrib.sampling.sampler._CAPI_NeighborSampling(
                self.graphs[graph_idx]._graph,
                dgl.utils.toindex([node_idx]).todgltensor(),
                0,  # batch_start_id
                1,  # batch_size
                1,  # workers
                self.num_neighbors,  # expand_factor
                self.rw_hops,  # num_hops
                "out",
                False,
                prob,
            )[0]
            nf1 = NodeFlow(self.graphs[graph_idx], nf1)
            trace1 = [nf1.layer_parent_nid(i) for i in range(nf1.num_layers)]
            nf2 = dgl.contrib.sampling.sampler._CAPI_NeighborSampling(
                self.graphs[graph_idx]._graph,
                dgl.utils.toindex([other_node_idx]).todgltensor(),
                0,  # batch_start_id
                1,  # batch_size
                1,  # workers
                self.num_neighbors,  # expand_factor
                self.rw_hops,  # num_hops
                "out",
                False,
                prob,
            )[0]
            nf2 = NodeFlow(self.graphs[graph_idx], nf2)
            trace2 = [nf2.layer_parent_nid(i) for i in range(nf2.num_layers)]
            traces = [trace1, trace2]
        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        graph_k = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            positional_embedding_size=self.positional_embedding_size,
        )
        if self.graph_transform:
            graph_q = self.graph_transform(graph_q)
            graph_k = self.graph_transform(graph_k)
        # return graph_q, graph_k
        # degree_list = []
        # start = self.dgl_graphs_file.rindex('_')
        # name = self.dgl_graphs_file[start + 1:-4]
        # with open("sample_degree_list_" + name + ".txt", "a") as fp:
        #     dgl_graph = dgl.to_bidirected(graph_q)
        #     num_nodes, num_edges = dgl_graph.batch_num_nodes[0], dgl_graph.batch_num_edges[0]
        #     degrees = dgl_graph.in_degrees(range(num_nodes))
        #     for key, value in enumerate(degrees):
        #         degree_list.append(value.item())
        #     for i in range(len(degree_list)):
        #         fp.write(str(degree_list[i]) + "\n")
        #     # nx_graph = dgl.to_networkx(dgl_graph)
        # degree_list = self.graphs[0].in_degrees().tolist()
        # return graph_q, graph_k, degree_list[node_idx]
        return graph_q, graph_k


class GraphDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
    ):
        super(GraphDataset).__init__()
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert sum(step_dist) == 1.0
        assert positional_embedding_size > 1
        #  graphs = []
        graphs, _ = dgl.data.utils.load_graphs(
            "data_bin/dgl/lscc_graphs.bin", [0, 1, 2]
        )
        for name in ["cs", "physics"]:
            g = Coauthor(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        for name in ["computers", "photo"]:
            g = AmazonCoBuy(name)[0]
            g.remove_nodes((g.in_degrees() == 0).nonzero().squeeze())
            g.readonly()
            graphs.append(g)
        # more graphs are comming ...
        print("load graph done")
        self.graphs = graphs
        self.length = sum([g.number_of_nodes() for g in self.graphs])

    def __len__(self):
        return self.length

    def _convert_idx(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()
        return graph_idx, node_idx

    def __getitem__(self, idx):
        graph_idx, node_idx = self._convert_idx(idx)

        step = np.random.choice(len(self.step_dist), 1, p=self.step_dist)[0]
        if step == 0:
            other_node_idx = node_idx
        else:
            other_node_idx = dgl.contrib.sampling.random_walk(
                g=self.graphs[graph_idx], seeds=[node_idx], num_traces=1, num_hops=step
            )[0][0][-1].item()

        max_nodes_per_seed = max(
            self.rw_hops,
            int(
                (
                        self.graphs[graph_idx].out_degree(node_idx)
                        * math.e
                        / (math.e - 1)
                        / self.restart_prob
                )
                + 0.5
            ),
        )
        traces = dgl.contrib.sampling.random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx, other_node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=max_nodes_per_seed,
        )

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )
        graph_k = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=other_node_idx,
            trace=traces[1],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=hasattr(self, "entire_graph") and self.entire_graph,
        )
        return graph_q, graph_k


class NodeClassificationDataset(GraphDataset):
    def __init__(
            self,
            dataset,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
    ):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        assert positional_embedding_size > 1

        self.data = data_util.create_node_classification_dataset(dataset).data
        # self.data = data_util.create_node_classification_dataset(dataset, flip=True, rate=0.1, train_flag=True).data
        self.graphs = [self._create_dgl_graph(self.data)]
        self.length = sum([g.number_of_nodes() for g in self.graphs])
        self.total = self.length

    def _create_dgl_graph(self, data):
        graph = dgl.DGLGraph()
        src, dst = data.edge_index.tolist()
        num_nodes = data.edge_index.max() + 1
        graph.add_nodes(num_nodes)
        graph.add_edges(src, dst)
        graph.add_edges(dst, src)
        graph.readonly()
        return graph


class GraphClassificationDataset(NodeClassificationDataset):
    def __init__(
            self,
            dataset,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
    ):
        self.rw_hops = rw_hops
        self.subgraph_size = subgraph_size
        self.restart_prob = restart_prob
        self.positional_embedding_size = positional_embedding_size
        self.step_dist = step_dist
        self.entire_graph = True
        assert positional_embedding_size > 1

        self.dataset = data_util.create_graph_classification_dataset(dataset)
        self.graphs = self.dataset.graph_lists

        self.length = len(self.graphs)
        self.total = self.length

    def _convert_idx(self, idx):
        graph_idx = idx
        node_idx = self.graphs[idx].out_degrees().argmax().item()
        return graph_idx, node_idx


class GraphClassificationDatasetLabeled(GraphClassificationDataset):
    def __init__(
            self,
            dataset,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
    ):
        super(GraphClassificationDatasetLabeled, self).__init__(
            dataset,
            rw_hops,
            subgraph_size,
            restart_prob,
            positional_embedding_size,
            step_dist,
        )
        self.num_classes = self.dataset.num_labels
        self.entire_graph = True
        self.dict = [self.getitem(idx) for idx in range(len(self))]

    def __getitem__(self, idx):
        return self.dict[idx]

    def getitem(self, idx):
        graph_idx = idx
        node_idx = self.graphs[idx].out_degrees().argmax().item()

        traces = dgl.contrib.sampling.random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=self.rw_hops,
        )

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
            entire_graph=True,
        )
        return graph_q, self.dataset.graph_labels[graph_idx].item()


class NodeClassificationDatasetLabeled(NodeClassificationDataset):
    def __init__(
            self,
            dataset,
            rw_hops=64,
            subgraph_size=64,
            restart_prob=0.8,
            positional_embedding_size=32,
            step_dist=[1.0, 0.0, 0.0],
            cat_prone=False,
    ):
        super(NodeClassificationDatasetLabeled, self).__init__(
            dataset,
            rw_hops,
            subgraph_size,
            restart_prob,
            positional_embedding_size,
            step_dist,
        )
        assert len(self.graphs) == 1
        self.num_classes = self.data.y.shape[1]

    def __getitem__(self, idx):
        graph_idx = 0
        node_idx = idx
        for i in range(len(self.graphs)):
            if node_idx < self.graphs[i].number_of_nodes():
                graph_idx = i
                break
            else:
                node_idx -= self.graphs[i].number_of_nodes()

        traces = dgl.contrib.sampling.random_walk_with_restart(
            self.graphs[graph_idx],
            seeds=[node_idx],
            restart_prob=self.restart_prob,
            max_nodes_per_seed=self.rw_hops,
        )

        graph_q = data_util._rwr_trace_to_dgl_graph(
            g=self.graphs[graph_idx],
            seed=node_idx,
            trace=traces[0],
            positional_embedding_size=self.positional_embedding_size,
        )
        return graph_q, self.data.y[idx].argmax().item()


if __name__ == "__main__":
    num_workers = 1
    import psutil

    mem = psutil.virtual_memory()
    print(mem.used / 1024 ** 3)
    graph_dataset = LoadBalanceGraphDataset(
        num_workers=num_workers, aug="ns", rw_hops=4, num_neighbors=5
    )
    mem = psutil.virtual_memory()
    print(mem.used / 1024 ** 3)
    graph_loader = torch.utils.data.DataLoader(
        graph_dataset,
        batch_size=1,
        collate_fn=data_util.batcher(),
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )
    mem = psutil.virtual_memory()
    print(mem.used / 1024 ** 3)
    for step, batch in enumerate(graph_loader):
        print("bs", batch[0].batch_size)
        print("n=", batch[0].number_of_nodes())
        print("m=", batch[0].number_of_edges())
        mem = psutil.virtual_memory()
        print(mem.used / 1024 ** 3)
        print(batch[0].ndata["pos_undirected"])
    exit(0)
    graph_dataset = NodeClassificationDataset(dataset="wikipedia")
    graph_loader = torch.utils.data.DataLoader(
        dataset=graph_dataset,
        batch_size=20,
        collate_fn=data_util.batcher(),
        shuffle=True,
        num_workers=4,
    )
    for step, batch in enumerate(graph_loader):
        print(batch.graph_q)
        print(batch.graph_q.ndata["x"].shape)
        print(batch.graph_q.batch_size)
        print("max", batch.graph_q.edata["efeat"].max())
        break
