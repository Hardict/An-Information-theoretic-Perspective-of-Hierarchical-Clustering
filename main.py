import time

import numpy as np
import networkx as nx
import math
import heapq
import queue
import copy
from sklearn import metrics
import sys

from typing import Set, List


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

from PartitionTree import PartitionTreeNode


# 二叉生成过程

def CalculateCutValue(A: dict, B: dict) -> float:
    """
    :param A:点A集合(节点id和邻接表{id,E})->dict
    :param B:点B集合(节点id和邻接表{id,E})->dict
    :return:
    """
    cutAB = 0
    for a in A.keys():
        for to in A[a]:
            if to in B.keys():
                if 'weight' in A[a][to]:
                    cutAB += A[a][to]['weight']
                else:
                    cutAB += 1
    return cutAB


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
    cutAB = CalculateCutValue(A.node_set, B.node_set)
    new_node.g_val = A.g_val + B.g_val - 2 * cutAB
    new_node.cut_val = cutAB
    new_node.height = max(A.height, B.height) + 1
    new_node.node_set = {**A.node_set, **B.node_set}  # >=python3.5
    new_node.origin_node_set = {*A.origin_node_set, *B.origin_node_set}  # >=python3.5
    A.parent = new_node
    B.parent = new_node
    return new_node


def CalculateMergeDelta(A: PartitionTreeNode, B: PartitionTreeNode, volumeV: float) -> float:
    """
    :param A:树节点A
    :param B:树节点B
    :param volumeV:图的体积
    :return:合并A，B后cost变化值
    """
    cutAB = CalculateCutValue(A.node_set, B.node_set)
    volumeC = A.volume + B.volume
    ret = 2 * cutAB / volumeV * np.log2(volumeC / volumeV)
    return ret


def BuildBinaryTree(G: nx.Graph, G_volume: float = None, hyperGraphHash: dict = None,
                    originchildrenHash: dict = None, volumeHash: dict = None, type: str = "SE") -> PartitionTreeNode:
    """
    :param G: 图G
    :param G_volume: 图G体积
    :param hyperGraphHash 超节点映射
    :param originchildrenHash 超节点孩子映射
    :return:划分树的根
    """
    # 初始建立num_vertex个树上节点
    # 因为适配子图不用nx.volume
    num_vertex = len(G.nodes)
    tree_node_list: List[PartitionTreeNode] = []
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
        if volumeHash is not None:
            # if volume != volumeHash[i]:
            #     print(i, volume, volumeHash[i])
            #     if hyperGraphHash is not None:
            #         print(hyperGraphHash[i])
            volume = volumeHash[i]
        volumeV += volume
        new_node.volume = volume
        new_node.g_val = volume
        new_node.height = 1
        new_node.node_set = {i: G.adj[i]}
        if hyperGraphHash is not None:
            new_node.origin_node_set = hyperGraphHash[i]
        else:
            new_node.origin_node_set = {i}
        if originchildrenHash is not None:
            new_node.origin_children = originchildrenHash[i]
        else:
            new_node.origin_children = None
        tree_node_list.append(new_node)
    if G_volume is not None:
        volumeV = G_volume
    # 最小堆
    min_heap: List[(float, PartitionTreeNode, PartitionTreeNode)] = []
    # 初始化delta变化

    # SE
    for i in range(num_vertex):
        for j in range(i + 1, num_vertex):
            A = tree_node_list[i]
            B = tree_node_list[j]
            if type == "SE":
                val = CalculateMergeDelta(A, B, volumeV)
            elif type == "single-linkage":
                val = float("inf")
                for a in A.node_set:
                    for b in B.node_set:
                        if G.has_edge(a, b):
                            if "weight" in G[a][b].keys():
                                val = min(val, G[a][b]["weight"])
                            else:
                                val = min(val, 1)
                if val == float("inf"):
                    val = 0
                val = -val
            elif type == "complete-linkage":
                val = -float("inf")
                for a in A.node_set:
                    for b in B.node_set:
                        if G.has_edge(a, b):
                            if "weight" in G[a][b].keys():
                                val = max(val, G[a][b]["weight"])
                            else:
                                val = max(val, 1)
                val = -val
            elif type == "average-linkage":
                cutAB = CalculateCutValue(A.node_set, B.node_set)
                val = cutAB / (len(A.node_set) * len(B.node_set))
                val = -val
            else:
                raise ValueError("type not define")
            heapq.heappush(min_heap, (val, A, B))

    vis = dict()  # 表示当前节点是否被合并
    for p in tree_node_list:
        vis[p.id] = False
    for t in range(num_vertex - 1):
        # num_vertex - 1次合并
        delta, A, B = heapq.heappop(min_heap)
        # 堆中可能有已经涉及合并的节点对，需要剔除
        while (vis[A.id] is True) or (vis[B.id] is True):
            delta, A, B = heapq.heappop(min_heap)
        # 合并A和B,得到节点C
        vis[A.id] = vis[B.id] = True
        C = MergePartitionTreeNode(A, B, next(id_generator))
        # wprint(C.id, A, B, delta)
        vis[C.id] = False
        for node in tree_node_list:
            if vis[node.id] is False:
                if type == "SE":
                    val = CalculateMergeDelta(node, C, volumeV)
                elif type == "single-linkage":
                    val = float("inf")
                    for a in node.node_set:
                        for b in C.node_set:
                            if G.has_edge(a, b):
                                if "weight" in G[a][b].keys():
                                    val = min(val, G[a][b]["weight"])
                                else:
                                    val = min(val, 1)
                    if val == float("inf"):
                        val = 0
                    val = -val
                elif type == "complete-linkage":
                    val = -float("inf")
                    for a in node.node_set:
                        for b in C.node_set:
                            if G.has_edge(a, b):
                                if "weight" in G[a][b].keys():
                                    val = max(val, G[a][b]["weight"])
                                else:
                                    val = max(val, 1)
                    val = -val
                elif type == "average-linkage":
                    cutAB = CalculateCutValue(node.node_set, C.node_set)
                    val = cutAB / (len(node.node_set) * len(C.node_set))
                    val = -val
                else:
                    raise ValueError("type not define")
                heapq.heappush(min_heap, (val, node, C))
        tree_node_list.append(C)
    # print(len(tree_node_list))
    return tree_node_list[-1]


# 压缩过程

def CompressPartitionTreeNode(A: PartitionTreeNode):
    """
    :param A:待压缩节点A
    :return:
    """
    if A.children is not None:
        for ch in A.children:
            ch.parent = A.parent
        A.parent.children = A.parent.children.union(A.children)
    A.parent.cut_val += A.cut_val  # 这里已经有了A与A.parent其它孩子的cut，实际新增cut是A.children作为A.parent孩子新增cut，实际就是A.cut
    # 自下而上更新height
    p = A.parent
    A.parent.children.remove(A)
    while p is not None:
        h = 0
        for ch in p.children:
            h = max(h, ch.height + 1)
        if h == p.height:
            break
        p.height = h
        p = p.parent


def CalculateCompressDelta(A: PartitionTreeNode, volumeV: float) -> float:
    """
    :param A: 树节点A
    :param volumeV:图的体积
    :return: A压缩后cost变化
    """
    g = A.cut_val
    return g / volumeV * np.log2(A.parent.volume / A.volume)


def CompressPartitionTree(root: PartitionTreeNode, target_height: int = 2, G_volume: float = None) -> PartitionTreeNode:
    """
    :param root: 待压缩的树的根节点
    :param target_height: 树的目标高度，默认为2
    :param G_volume: 图的体积
    :return: 压缩后的树的根节点
    """
    # print(root)
    # 初始化计算所有节点
    min_heap: List[(float, PartitionTreeNode, int)] = []
    Q: queue.Queue(PartitionTreeNode) = queue.Queue()
    Q.put(root)
    # 因为压缩操作会改变节点父亲的Delta值，利用一个dict记录最后更新时间
    delta_valid_time: dict[int:int] = dict()
    time_stamp: int = 0
    if G_volume is None:
        G_volume = root.volume
    while not Q.empty():
        p = Q.get()
        if p.children is not None:
            if p.id != root.id:
                val = CalculateCompressDelta(p, G_volume)
                heapq.heappush(min_heap, (val, p, time_stamp))
                delta_valid_time[p.id] = time_stamp
            for ch in p.children:
                Q.put(ch)
    while root.height > target_height:
        # print([(ch.id, ch.height) for ch in root.children])
        val, p, update_time = heapq.heappop(min_heap)
        while delta_valid_time[p.id] != update_time:
            val, p, update_time = heapq.heappop(min_heap)
        time_stamp += 1
        pp = p.parent  # 这里pp防止歧义
        CompressPartitionTreeNode(p)
        # 自身cut值改变需要更新
        if pp.id != root.id:
            val = CalculateCompressDelta(pp, G_volume)
            heapq.heappush(min_heap, (val, pp, time_stamp))
            delta_valid_time[pp.id] = time_stamp
        # 部分节点父亲节点改变需要更新
        for ch in p.children:
            if ch.children is not None:
                val = CalculateCompressDelta(ch, G_volume)
                heapq.heappush(min_heap, (val, ch, time_stamp))
                delta_valid_time[ch.id] = time_stamp
    return root


# HCSE

def CalculateEntropy(root: PartitionTreeNode, G_volume: float = None) -> float:
    """
    :param root: 划分树的根
    :param G_volume: 图的体积
    :return: 划分树的熵
    """
    ent = 0
    Q: queue.Queue(PartitionTreeNode) = queue.Queue()
    Q.put(root)
    if G_volume is None:
        G_volume = root.volume
    while not Q.empty():
        p = Q.get()
        # 这里因为涉及子树最好不要用p.parent来判断
        if p.children is not None:
            for ch in p.children:
                ent += -(ch.g_val / G_volume) * np.log2(ch.volume / p.volume)
                Q.put(ch)
    return ent


def hypergraph(G: nx.Graph, node2community) -> nx.Graph:
    new_G = nx.Graph()
    # flag = False
    for u, v, data in G.edges(data=True):
        # 防止孤立点消失
        if u in node2community and v in node2community:
            # if int(u) < 100:
            #     flag = True
            c_u = node2community[u]  # 获取节点 u 所属的社区
            c_v = node2community[v]  # 获取节点 v 所属的社区
            # print(c_u, c_v)
            if c_u != c_v or u == v:
                # 社区 c_u 到社区 c_v 之间的连边数量加1，如果边是带权边则可以累加权重
                if new_G.has_edge(c_u, c_v):
                    new_G[c_u][c_v]['weight'] += G[u][v]['weight']
                else:
                    new_G.add_edge(c_u, c_v, weight=G[u][v]['weight'])
            else:
                if new_G.has_edge(c_u, c_v):
                    new_G[c_u][c_v]['weight'] += 2 * G[u][v]['weight']
                else:
                    new_G.add_edge(c_u, c_v, weight=2 * G[u][v]['weight'])
        elif u in node2community:
            new_G.add_node(node2community[u])
        elif v in node2community:
            new_G.add_node(node2community[v])
    # if flag:
    #     with open('subgraph.in', 'w') as f:
    #         for u, v, data in new_G.edges(data=True):
    #             f.write(str(u) + " " + str(v) + "\n")
    return new_G


def super_node_dfs_update(u: PartitionTreeNode):
    """
    更新超节点信息
    :param u:
    :return:
    """
    if u.origin_children is not None:
        u.children = u.origin_children
        if u.children is not None:
            for v in u.children:
                u.height = max(u.height, v.height + 1)
                v.parent = u
        u.origin_children = None
        return
    if u.children is not None:
        for v in u.children:
            super_node_dfs_update(v)
            u.height = max(u.height, v.height + 1)


def HCSE(G: nx.Graph, target_height: int = 2, type: str = "SE", delta_fp=None) -> PartitionTreeNode:
    """
    :param G: networkx格式的图G
    :param target_height: 树的目标高度（层次数），默认为2
    :return: 划分树的根
    """
    G_volume = nx.volume(G, G.nodes)
    top_root = BuildBinaryTree(G, G_volume, type=type)
    top_root = CompressPartitionTree(root=top_root, target_height=3, G_volume=G_volume)
    deltaH_pre = 0
    H0 = 0
    volumeV = 0
    for i in G.nodes:
        volume = 0
        for to in G.adj[i]:
            if 'weight' in G.adj[i][to]:
                volume += G.adj[i][to]['weight']
            else:
                volume += 1
        H0 += -volume / G_volume * np.log2(volume / G_volume)
    deltaH = CalculateEntropy(top_root, G_volume) - H0
    if delta_fp is not None:
        print("H0", H0, file=fp)
        print(-deltaH, file=delta_fp)
        print(-deltaH / H0, file=delta_fp)
    deltaH_pre = deltaH
    while top_root.height < target_height:
        # print("???", top_root.height, file=fp)
        # Q: queue.Queue(PartitionTreeNode) = queue.Queue()
        # Q.put(top_root)
        # while not Q.empty():
        #     p = Q.get()
        #     print(p, file=fp)
        #     if p.children is not None:
        #         for ch in p.children:
        #             Q.put(ch)
        # 选一层进行扩展
        extend_node: List[PartitionTreeNode] = []
        # 按高度分层
        Q_list: List[queue.Queue(PartitionTreeNode)] = [queue.Queue(), queue.Queue()]
        layer_node_list: List[List[PartitionTreeNode]] = []
        tmp_list: List[PartitionTreeNode] = []
        op = 0
        Q_list[op].put(top_root)
        while not Q_list[op].empty():
            p = Q_list[op].get()
            tmp_list.append(p)
            if p.children is not None:
                for ch in p.children:
                    if ch.height != 1:
                        Q_list[op ^ 1].put(ch)
            if Q_list[op].empty():
                layer_node_list.append(tmp_list)
                tmp_list = []
                op ^= 1
        # Todo: 选择一层
        delta_max = 1
        extend_layer_id = 0
        kkk = 0
        for layer in layer_node_list:
            delta = 0
            for p in layer:
                node2community = dict()
                volumeHash = dict()
                for ch in p.children:
                    for u in ch.origin_node_set:
                        node2community[u] = ch.id
                    volumeHash[ch.id] = ch.volume
                # print(p.origin_node_set)
                # print(node2community)
                subgraph = hypergraph(G, node2community)
                new_p = BuildBinaryTree(subgraph, volumeHash=volumeHash, type=type)
                new_p = CompressPartitionTree(root=new_p, target_height=3)
                delta -= (CalculateEntropy(new_p, G_volume) - CalculateEntropy(p, G_volume)) / CalculateEntropy(p,
                                                                                                                G_volume)
            print(delta, len(layer), delta / len(layer))
            delta /= len(layer)
            if delta < delta_max:
                delta_max = delta
                extend_node = layer
                extend_layer_id = kkk
            kkk += 1
        # 自适应
        if delta_max <= 0:
            break
        # 进行扩展
        # print(len(extend_node))
        H0 = CalculateEntropy(top_root, G_volume)
        for p in extend_node:
            node2community = dict()
            hyperGraphHash = dict()
            originchildrenHash = dict()
            volumeHash = dict()
            for ch in p.children:
                hyperGraphHash[ch.id] = set()
                originchildrenHash[ch.id] = ch.children
                for u in ch.origin_node_set:
                    node2community[u] = ch.id
                    hyperGraphHash[ch.id].add(u)
                volumeHash[ch.id] = ch.volume
            # print(sorted(list(node2community.keys())))
            # print(node2community)
            subgraph = hypergraph(G, node2community)
            new_p = BuildBinaryTree(subgraph, hyperGraphHash=hyperGraphHash, originchildrenHash=originchildrenHash,
                                    volumeHash=volumeHash, type=type)
            new_p = CompressPartitionTree(root=new_p, target_height=3)
            # print(node2community, file=fp)
            # print(subgraph.nodes, file=fp)
            # print(new_p, file=fp)
            # print(len(list(ch.id for ch in new_p.children)))
            # 更新超节点下挂节点
            super_node_dfs_update(new_p)
            # print(new_p, file=fp)
            new_p.parent = p.parent
            if p.parent is not None:
                origin_p = p
                origin_p.parent.children.add(new_p)
                # 高度更新
                while (p.parent is not None):
                    h = 0
                    for ch in p.parent.children:
                        h = max(h, ch.height + 1)
                    p.parent.height = h
                    p = p.parent
                origin_p.parent.children.remove(origin_p)
            else:
                top_root = new_p
        deltaH = CalculateEntropy(top_root, G_volume) - H0
        if delta_fp is not None:
            print("H0", H0, file=fp)
            print(-deltaH, file=delta_fp)
            print(delta_max, file=delta_fp)
            print("extend_layer_id: ", extend_layer_id, file=delta_fp)

    return top_root


def BalanceTree(root: PartitionTreeNode) -> PartitionTreeNode:
    """
    将树变为等高（完全叉树）
    :param root:树的根
    :return:
    """
    Q: queue.Queue(PartitionTreeNode) = queue.Queue()
    Q.put(root)
    while not Q.empty():
        p = Q.get()
        if p.parent is not None:
            x = p
            while x.parent.height > x.height + 1:
                # 扩展p
                new_x: PartitionTreeNode = PartitionTreeNode()
                new_x.parent = x.parent
                new_x.height = x.height + 1
                new_x.id = next(id_generator)
                new_x.children = {x}
                new_x.volume = x.volume
                new_x.origin_node_set = x.origin_node_set
                new_x.node_set = x.node_set
                new_x.g_val = x.g_val
                new_x.cut_val = 0
                x.parent.children.remove(x)
                x.parent.children.add(new_x)
                x.parent = new_x
                x = new_x
        if p.children is not None:
            for ch in p.children:
                Q.put(ch)
    return root


import newHSBM
import nmi
import vis_tree
import cmp_louvain
import cmp_hlp


def CalculateNMI(TestLayers, GroundTruthLayers, fp=None):
    for t in range(1, len(TestLayers)):
        val = 0
        k0 = t
        print(len(TestLayers[t]), t, file=fp)
        # S = set(TestLayers[t].values())
        # print("Test:", len(S), S, file=fp)
        for k in range(1, len(GroundTruthLayers)):
            cluster_ground_truth = []
            cluster_test = []
            sort_key = sorted(list(GroundTruthLayers[k].keys()))

            for key in sort_key:
                cluster_test.append(TestLayers[t][key])
                cluster_ground_truth.append(GroundTruthLayers[k][key])
            tmp = nmi.calc_nmi(cluster_test, cluster_ground_truth)
            if tmp > val:
                val = tmp
                k0 = k
            print("testLayer={},groundtruthLayer={},nmi={}".format(t, k, tmp), file=fp)
        # S = set(GroundTruthLayers[k0].values())
        # print("GT:", len(S), S, file=fp)
        # print("testLayer={},groundtruthLayer={},nmi={}".format(t, k0, val), file=fp)


if __name__ == '__main__':
    layer_op = eval(sys.argv[1])
    if layer_op == 3:
        # 3层
        params = [(10, 1, 1), (100, 10, 1), (1000, 100, 1)]
        probs_list = []
        probs_list.append([])
        probs_list.append([1.5e-4, 6.5e-2, 0.9])
        probs_list.append([2.5e-4, 4.5e-2, 0.9])
        probs_list.append([3.5e-4, 7.5e-2, 0.9])
        probs = probs_list[int(sys.argv[2])]

        fp = open(
            "final\\final-1000-{}-{}-{}.txt".format(sys.argv[2], sys.argv[3],
                                                    time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())),
            mode="w", encoding="utf-8")

    elif layer_op == 4:
        # 4层
        params = [(5, 1, 3), (25, 5, 3), (250, 25, 5), (2500, 250, 5)]
        probs_list = []
        probs_list.append([])
        probs_list.append([6e-6, 1.5e-3, 4.5e-2, 9e-1])
        probs_list.append([4e-6, 1.5e-3, 5.5e-2, 9e-1])
        probs_list.append([2.5e-6, 7.5e-4, 4.5e-2, 9e-1])
        probs = probs_list[int(sys.argv[2])]
        fp = open(
            "final\\final-2500-{}-{}-{}.txt".format(sys.argv[2], sys.argv[3],
                                                       time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())),
            mode="w", encoding="utf-8")
    elif layer_op == 5:
        params = [(5, 1, 5), (25, 5, 5), (125, 25, 5), (600, 125, 4), (6000, 600, 10)]
        probs_list = []
        probs_list.append([])
        probs_list.append([1e-6, 7.5e-5, 1.5e-3, 1.5e-2, 9e-1])
        probs_list.append([6e-7, 7.5e-5, 4.5e-3, 4.5e-2, 9e-1])
        probs_list.append([3e-7, 5e-5, 1.5e-3, 1.5e-2, 9e-1])
        probs = probs_list[int(sys.argv[2])]
        fp = open(
            "final\\final-6000-{}-{}-{}.txt".format(sys.argv[2], sys.argv[3],
                                                       time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())),
            mode="w", encoding="utf-8")
    noweighted_G, tree = newHSBM.generate_HSBM(params, probs)
    print("size={}".format(params), file=fp)
    print("probs={}".format(probs), file=fp)
    while len(noweighted_G.nodes) != params[-1][0]:
        print(noweighted_G)
        noweighted_G, tree = newHSBM.generate_HSBM(params, probs)
    # with open('graph.in', 'w') as f:
    #     for u, v in noweighted_G.edges():
    #         f.write(str(u) + " " + str(v) + "\n")

    # noweighted_G: nx.Graph = nx.read_edgelist("subgraph.in", nodetype=str, data=False)
    G = nx.Graph()
    # G.add_weighted_edges_from([(str(u), str(v), 1) for u, v in noweighted_G.edges()])
    edge_list = [(str(u), str(v), 1) for u, v in noweighted_G.edges()]
    np.random.shuffle(edge_list)
    G.add_weighted_edges_from(edge_list)
    print(G)
    Q: queue.Queue(PartitionTreeNode) = queue.Queue()
    ground_layer_class = dict()
    for k in range(len(probs) + 1):
        # k 是第k层，根结点为第0层
        dic = dict()
        clss = tree.get_k_layer_ground_truth(tree.root, k)
        for key in clss.keys():
            for nd in clss[key]:
                dic[nd] = key
        ground_layer_class[k] = dic

    type_list = ["SE", "single-linkage", "complete-linkage", "average-linkage"]
    type_list = ["SE"]
    for type in type_list:
        id_generator = NewIDPartitionTreeNode(len(G.nodes))
        print("HCSE type:{}".format(type), file=fp)
        if type == "SE":
            root = HCSE(G, len(probs) + 1, type=type, delta_fp=fp)
        else:
            newG = copy.deepcopy(G)
            for (u, v) in newG.edges(data=False):
                newG[u][v]['weight'] += np.random.rand() / 1e6
            root = HCSE(newG, len(probs) + 1, type=type)
        root = BalanceTree(root)

        Q_list = [queue.Queue(), queue.Queue()]
        op = 0
        Q_list[op].put(root)
        tmp_list = []
        layer_class = dict()
        cnt_layer = 0
        while not Q_list[op].empty():
            p = Q_list[op].get()
            tmp_list.append(p)
            if p.children is not None:
                for ch in p.children:
                    Q_list[op ^ 1].put(ch)
            if Q_list[op].empty():
                dic = dict()
                for node in tmp_list:
                    for origin_node in node.origin_node_set:
                        dic[origin_node] = node.id
                layer_class[cnt_layer] = dic
                cnt_layer += 1
                tmp_list = []
                op ^= 1

        print("HCSE type:{}".format(type), file=fp)
        CalculateNMI(layer_class, ground_layer_class, fp)
    # exit(0)

    print("louvain", file=fp)
    louvain_layers = cmp_louvain.LouvainLayers(G)
    # print(louvain_layers, file=fp)
    root = vis_tree.build(louvain_layers)
    vis_tree.dfs(root, G)
    louvain_layer_class = dict()
    for k in range(len(louvain_layers)):
        # k 是第k层，根结点为第0层
        dic = dict()
        clss = vis_tree.get_k_layer_ground_truth(root, k)
        # print(clss, file=fp)
        for key in clss.keys():
            for nd in clss[key]:
                dic[nd] = key
        louvain_layer_class[k] = dic
    CalculateNMI(louvain_layer_class, ground_layer_class, fp)
    # print(SE_val, louvain_val)

    print("HLP", file=fp)
    hlp_layers = cmp_hlp.HLPLayers(G)
    print(hlp_layers, file=fp)
    root = vis_tree.build(hlp_layers)
    vis_tree.dfs(root, G)
    hlp_layer_class = dict()
    for k in range(len(hlp_layers)):
        # k 是第k层，根结点为第0层
        dic = dict()
        clss = vis_tree.get_k_layer_ground_truth(root, k)
        print(clss, file=fp)
        for key in clss.keys():
            for nd in clss[key]:
                dic[nd] = key
        hlp_layer_class[k] = dic
    CalculateNMI(hlp_layer_class, ground_layer_class, fp)
    # exit(0)
