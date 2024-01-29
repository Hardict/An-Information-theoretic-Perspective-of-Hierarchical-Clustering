import copy

import numpy as np
import scipy as sp
import networkx as nx
import heapq

import linkagePlusPlus
from typing import List

from PartitionTree import PartitionTreeNode
import queue

import sys

sys.path.append(
    r"hierarchical-clustering-well-clustered-graphs-main")
from Tree_Construction import agg_constr


def NewIDPartitionTreeNode(start_val: int = 1):
    """
    利用yield分配Node的ID
    :return:
    """
    i = start_val
    while True:
        yield i
        i += 1


id_generator = 1


def HuffmanMerge(G: nx.Graph) -> PartitionTreeNode:
    num_vertex = len(G.nodes)
    tree_node_list: list[PartitionTreeNode] = []
    volumeV = 0
    for i in G.nodes:
        new_node = PartitionTreeNode()
        new_node.id = next(id_generator)
        volume = 0
        for to in G.adj[i]:
            if 'weight' in G.adj[i][to]:
                volume += G.adj[i][to]['weight']
            else:
                volume += 1
        volumeV += volume
        new_node.volume = volume
        new_node.height = 1
        new_node.node_set = {i: G.adj[i]}
        new_node.origin_node_set = {i}
        tree_node_list.append(new_node)
    return SubHuffmanMerge(tree_node_list)


def SubHuffmanMerge(tree_node_list: List[PartitionTreeNode]) -> PartitionTreeNode:
    min_heap: list[(float, PartitionTreeNode)] = []
    # 初始化delta变化
    num_node = len(tree_node_list)
    for i in range(num_node):
        A = tree_node_list[i]
        heapq.heappush(min_heap, (A.volume, A))

    for t in range(num_node - 1):
        # num_node - 1次合并
        volumeA, A = heapq.heappop(min_heap)
        volumeB, B = heapq.heappop(min_heap)
        C = MergePartitionTreeNode(A, B, next(id_generator))
        heapq.heappush(min_heap, (C.volume, C))
        tree_node_list.append(C)
    return tree_node_list[-1]


def MergePartitionTreeNode(A: PartitionTreeNode, B: PartitionTreeNode, new_node_id: int = 0) -> PartitionTreeNode:
    """
    :param A:待合并节点A
    :param B:待合并节点B
    :param new_node_id: 新节点id
    :return:新节点
    """
    new_node = PartitionTreeNode(_parent=None, _children={A, B})
    if new_node_id != 0:
        new_node.id = new_node_id
    new_node.volume = A.volume + B.volume
    new_node.height = max(A.height, B.height) + 1
    new_node.node_set = {**A.node_set, **B.node_set}  # >=python3.5
    new_node.origin_node_set = {*A.origin_node_set, *B.origin_node_set}  # >=python3.5
    A.parent = new_node
    B.parent = new_node
    return new_node


import partition_cut
import vis_tree

if __name__ == '__main__':
    # print(1)
    # G = nx.path_graph(10)
    p1_list = [1e-3, 5e-3, 10e-3, 50e-3, 100e-3, 200e-3]
    p1_list = [5e-3]
    for p1 in p1_list:
        num_test = 1
        data_dict = dict()
        method_list = ["HuffmanMerge", "average-linkage", "single-linkage", "complete-linkage", "linkage++"]
        for method in method_list:
            data_dict[method] = dict()
            data_dict[method]["Das"] = list()
            data_dict[method]["SE"] = list()
            data_dict[method]["equal SE"] = list()
            data_dict[method]["depth_balance_factor"] = list()
            data_dict[method]["size_balance_factor"] = list()
            data_dict[method]["volume_balance_factor"] = list()
        num_blocks = 5
        prob_clique = 9e-1
        prob_cluster = p1
        for t in range(num_test):
            prob_matrix = np.zeros((num_blocks, num_blocks))
            for i in range(num_blocks):
                for j in range(num_blocks):
                    if i == j:
                        prob_matrix[i][j] = prob_clique
                    else:
                        prob_matrix[i][j] = prob_cluster
            size_list = []
            offset = 4
            for i in range(num_blocks):
                size = 1 << (i + offset)
                if i == 0:
                    size <<= 1
                size += np.random.randint(-5, 5)
                size = np.random.randint(1 << (offset), 1 << (2 + offset))
                size_list.append(size)
            noweighted_G = nx.stochastic_block_model(size_list, prob_matrix)
            G = nx.Graph()
            edge_list = [(str(u), str(v), 1) for u, v in noweighted_G.edges()]
            np.random.shuffle(edge_list)
            G.add_weighted_edges_from(edge_list)
            clusters = partition_cut.compute_improved_partition(G, num_blocks)
            id_generator = NewIDPartitionTreeNode(len(G.nodes))
            subG_root_list: List[PartitionTreeNode] = []
            for cluster in clusters:
                # print(len(cluster), cluster)
                subG = nx.subgraph(G, cluster)
                root = HuffmanMerge(subG)
                subG_root_list.append(root)
            root = SubHuffmanMerge(subG_root_list)
            lis = []
            Q_list = [queue.Queue(), queue.Queue()]
            op = 0
            Q_list[op].put(root)
            tmp_list = []
            while not Q_list[op].empty():
                p = Q_list[op].get()
                tmp_list.append(p)
                if p.children is not None:
                    for ch in p.children:
                        Q_list[op ^ 1].put(ch)
                if Q_list[op].empty():
                    dic = dict()
                    for node in tmp_list:
                        if node.children is not None:
                            S = set()
                            for ch in node.children:
                                if ch.children is None:
                                    # 应该保证叶只有一个点
                                    S.add(list(ch.origin_node_set)[0])
                                else:
                                    S.add(ch.id)
                            dic[node.id] = S
                    lis.append(dic)
                    tmp_list = []
                    op ^= 1
            lis[0]['root'] = lis[0].pop(root.id)
            # print(lis)
            tree_root = vis_tree.build(lis)
            vis_tree.dfs(tree_root, G)
            volumeG = nx.volume(G, G.nodes)
            val = 0
            for u in G.nodes:
                volume = 0
                for to in G.adj[u]:
                    if 'weight' in G.adj[u][to]:
                        volume += G.adj[u][to]['weight']
                    else:
                        volume += 1
                if volume > 0:
                    val += volume * np.log2(volume)
            val = (vis_tree.calc_SE(tree_root, volumeG) * volumeG + val) / 2
            method = "HuffmanMerge"
            data_dict[method]["Das"].append(vis_tree.calc_Das(tree_root))
            data_dict[method]["SE"].append(vis_tree.calc_SE(tree_root, volumeG))
            data_dict[method]["equal SE"].append(val)
            data_dict[method]["depth_balance_factor"].append(
                vis_tree.calc_depth_balance_factor(vis_tree.get_leaf_depth(tree_root, depth=0)))
            data_dict[method]["size_balance_factor"].append(
                vis_tree.calc_size_balance_factor(tree_root) / vis_tree.get_internal_nodes_size_sum(tree_root))
            data_dict[method]["volume_balance_factor"].append(
                vis_tree.calc_volume_balance_factor(tree_root) / vis_tree.get_internal_nodes_volume_sum(tree_root))

            tree_al = agg_constr.build_agg_tree(G, 'average_linkage')
            sh_layers = tree_al.print_layers()
            tree_root = vis_tree.build(sh_layers)
            vis_tree.dfs(tree_root, G)
            val = 0
            for u in G.nodes:
                volume = 0
                for to in G.adj[u]:
                    if 'weight' in G.adj[u][to]:
                        volume += G.adj[u][to]['weight']
                    else:
                        volume += 1
                if volume > 0:
                    val += volume * np.log2(volume)
            val = (vis_tree.calc_SE(tree_root, volumeG) * volumeG + val) / 2
            method = "average-linkage"
            data_dict[method]["Das"].append(vis_tree.calc_Das(tree_root))
            data_dict[method]["SE"].append(vis_tree.calc_SE(tree_root, volumeG))
            data_dict[method]["equal SE"].append(val)
            data_dict[method]["depth_balance_factor"].append(
                vis_tree.calc_depth_balance_factor(vis_tree.get_leaf_depth(tree_root, depth=0)))
            data_dict[method]["size_balance_factor"].append(
                vis_tree.calc_size_balance_factor(tree_root) / vis_tree.get_internal_nodes_size_sum(tree_root))
            data_dict[method]["volume_balance_factor"].append(
                vis_tree.calc_volume_balance_factor(tree_root) / vis_tree.get_internal_nodes_volume_sum(tree_root))

            newG = copy.deepcopy(G)
            for (u, v) in newG.edges(data=False):
                newG[u][v]['weight'] += np.random.rand() / 1e6
            tree_sl = agg_constr.build_agg_tree(newG, 'single_linkage')
            sh_layers = tree_sl.print_layers()
            tree_root = vis_tree.build(sh_layers)
            vis_tree.dfs(tree_root, G)
            val = 0
            for u in G.nodes:
                volume = 0
                for to in G.adj[u]:
                    if 'weight' in G.adj[u][to]:
                        volume += G.adj[u][to]['weight']
                    else:
                        volume += 1
                if volume > 0:
                    val += volume * np.log2(volume)
            val = (vis_tree.calc_SE(tree_root, volumeG) * volumeG + val) / 2
            method = "single-linkage"
            data_dict[method]["Das"].append(vis_tree.calc_Das(tree_root))
            data_dict[method]["SE"].append(vis_tree.calc_SE(tree_root, volumeG))
            data_dict[method]["equal SE"].append(val)
            data_dict[method]["depth_balance_factor"].append(
                vis_tree.calc_depth_balance_factor(vis_tree.get_leaf_depth(tree_root, depth=0)))
            data_dict[method]["size_balance_factor"].append(
                vis_tree.calc_size_balance_factor(tree_root) / vis_tree.get_internal_nodes_size_sum(tree_root))
            data_dict[method]["volume_balance_factor"].append(
                vis_tree.calc_volume_balance_factor(tree_root) / vis_tree.get_internal_nodes_volume_sum(tree_root))

            tree_cl = agg_constr.build_agg_tree(newG, 'complete_linkage')
            cl_layers = tree_cl.print_layers()
            tree_root = vis_tree.build(cl_layers)
            vis_tree.dfs(tree_root, G)
            val = 0
            for u in G.nodes:
                volume = 0
                for to in G.adj[u]:
                    if 'weight' in G.adj[u][to]:
                        volume += G.adj[u][to]['weight']
                    else:
                        volume += 1
                if volume > 0:
                    val += volume * np.log2(volume)
            val = (vis_tree.calc_SE(tree_root, volumeG) * volumeG + val) / 2
            method = "complete-linkage"
            data_dict[method]["Das"].append(vis_tree.calc_Das(tree_root))
            data_dict[method]["SE"].append(vis_tree.calc_SE(tree_root, volumeG))
            data_dict[method]["equal SE"].append(val)
            data_dict[method]["depth_balance_factor"].append(
                vis_tree.calc_depth_balance_factor(vis_tree.get_leaf_depth(tree_root, depth=0)))
            data_dict[method]["size_balance_factor"].append(
                vis_tree.calc_size_balance_factor(tree_root) / vis_tree.get_internal_nodes_size_sum(tree_root))
            data_dict[method]["volume_balance_factor"].append(
                vis_tree.calc_volume_balance_factor(tree_root) / vis_tree.get_internal_nodes_volume_sum(tree_root))

            lpp_layers = linkagePlusPlus.GetLayers(linkagePlusPlus.LinkagePlusPlus(G, num_blocks))
            tree_root = vis_tree.build(lpp_layers)
            vis_tree.dfs(tree_root, G)
            val = 0
            for u in G.nodes:
                volume = 0
                for to in G.adj[u]:
                    if 'weight' in G.adj[u][to]:
                        volume += G.adj[u][to]['weight']
                    else:
                        volume += 1
                if volume > 0:
                    val += volume * np.log2(volume)
            val = (vis_tree.calc_SE(tree_root, volumeG) * volumeG + val) / 2
            method = "linkage++"
            data_dict[method]["Das"].append(vis_tree.calc_Das(tree_root))
            data_dict[method]["SE"].append(vis_tree.calc_SE(tree_root, volumeG))
            data_dict[method]["equal SE"].append(val)
            data_dict[method]["depth_balance_factor"].append(
                vis_tree.calc_depth_balance_factor(vis_tree.get_leaf_depth(tree_root, depth=0)))
            data_dict[method]["size_balance_factor"].append(
                vis_tree.calc_size_balance_factor(tree_root) / vis_tree.get_internal_nodes_size_sum(tree_root))
            data_dict[method]["volume_balance_factor"].append(
                vis_tree.calc_volume_balance_factor(tree_root) / vis_tree.get_internal_nodes_volume_sum(tree_root))

        fp = open(
            "BBM\\5-24-random-{}blocks-{}clique-{}cluster.txt".format(num_blocks, int(prob_clique * 1000),
                                                                                int(prob_cluster * 1000)),
            mode="w", encoding="utf-8")
        print("number of test:{}".format(num_test), file=fp)
        print("number of blocks: ", num_blocks, file=fp)
        print("prob of clique: ", prob_clique, file=fp)
        print("prob of cluster:", prob_cluster, file=fp)
        index_list = ["Das", "SE", "equal SE", "depth_balance_factor", "size_balance_factor", "volume_balance_factor"]
        for method in method_list:
            print(method, file=fp)
            for index in index_list:
                print("{}: ".format(index),
                      "mean: ", np.mean(data_dict[method][index]),
                      ", min:", np.min(data_dict[method][index]),
                      ", max:", np.max(data_dict[method][index]),
                      file=fp)
            print("data: ", file=fp)
            for index in index_list:
                print(data_dict[method][index], file=fp)
        print("=============", file=fp)
