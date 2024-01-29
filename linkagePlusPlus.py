import copy
import queue
import numpy as np
import scipy as sp
import networkx as nx
import heapq
import newHSBM
import graph_methods as graph
from PartitionTree import PartitionTreeNode
from typing import List


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


def LinkagePlusPlus(G: nx.Graph, k: int) -> PartitionTreeNode:
    A = np.zeros((len(G.nodes), len(G.nodes)))
    node2id = dict()
    id2node = dict()
    degree_dict = dict()
    for id, nd in enumerate(G.nodes):
        node2id[nd] = id
        id2node[id] = nd
        degree_dict[id] = 0
    for u in G.nodes():
        for v in G.nodes():
            if G.has_edge(u, v):
                if "weight" in G.adj[u][v]:
                    w = G[u][v]["weight"]
                else:
                    w = 1
                uid = node2id[u]
                vid = node2id[v]
                A[uid][vid] -= w
                A[uid][uid] += w
                degree_dict[uid] += w
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i][j] /= np.sqrt(degree_dict[i] * degree_dict[j])
    eigs, Uk = sp.sparse.linalg.eigsh(A, k=k, which="SM", return_eigenvectors=True)
    # print(eigs)
    # print(Uk)
    for i in range(len(G.nodes)):
        Uk[i] = Uk[i] / np.linalg.norm(Uk[i])
    # print(Uk)

    num_vertex = len(G.nodes)
    tree_node_list: List[PartitionTreeNode] = []
    volumeV = 0
    if 'id_generator' not in locals().keys() or isinstance(id_generator, int):
        id_generator = NewIDPartitionTreeNode(len(G.nodes))
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

    min_heap: List[(float, PartitionTreeNode, PartitionTreeNode)] = []
    dis_matrix = np.zeros((len(G.nodes), len(G.nodes)))
    for u in G.nodes:
        minid = -1
        for v in G.nodes:
            uid = node2id[u]
            vid = node2id[v]
            dis = np.linalg.norm(Uk[uid] - Uk[vid])
            dis_matrix[uid][vid] = dis
            if uid != vid:
                if minid == -1:
                    minid = vid
                elif dis_matrix[uid][vid] < dis_matrix[uid][minid]:
                    minid = vid
        # print(u,minid)
    # print(dis_matrix)
    for i in range(num_vertex):
        for j in range(i + 1, num_vertex):
            A = tree_node_list[i]
            B = tree_node_list[j]
            val = float("inf")
            for a in A.node_set:
                for b in B.node_set:
                    aid = node2id[a]
                    bid = node2id[b]
                    val = min(val, np.linalg.norm(Uk[aid] - Uk[bid]))
            if val == float("inf"):
                val = 0
            heapq.heappush(min_heap, (val, A, B))
    vis = dict()  # 表示当前节点是否被合并
    for p in tree_node_list:
        vis[p.id] = False
    for t in range(num_vertex - k):
        # num_vertex - k次合并
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
                val = float("inf")
                for a in node.node_set:
                    for b in C.node_set:
                        aid = node2id[a]
                        bid = node2id[b]
                        val = min(val, np.linalg.norm(Uk[aid] - Uk[bid]))
                if val == float("inf"):
                    val = 0
                heapq.heappush(min_heap, (val, node, C))
        tree_node_list.append(C)
    # print(len(tree_node_list))
    # for node in tree_node_list:
    #     print(node)
    unmerged_node_list = []
    for node in tree_node_list:
        if vis[node.id] is False:
            unmerged_node_list.append(node)
    min_heap = []
    sim_dict = dict()
    for i in range(len(unmerged_node_list)):
        for j in range(i + 1, len(unmerged_node_list)):
            A = unmerged_node_list[i]
            B = unmerged_node_list[j]
            sim = 0
            cutAB = 0
            for a in A.node_set.keys():
                for to in A.node_set[a]:
                    if to in B.node_set.keys():
                        if 'weight' in A.node_set[a][to]:
                            cutAB += A.node_set[a][to]['weight']
                        else:
                            cutAB += 1
            sim = cutAB / (len(A.node_set) * len(B.node_set))
            sim_dict[(A.id, B.id)] = sim
            sim_dict[(B.id, A.id)] = sim
            heapq.heappush(min_heap, (-sim, A, B))
    for t in range(num_vertex - k, num_vertex - 1):
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
                sim = 0
                sim = max(sim_dict[(A.id, node.id)], sim_dict[(B.id, node.id)])
                sim_dict[(C.id, node.id)] = sim_dict[(node.id, C.id)] = sim
                heapq.heappush(min_heap, (-sim, node, C))
        tree_node_list.append(C)
    return tree_node_list[-1]


def GetLayers(root: PartitionTreeNode):
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
    return lis


if __name__ == "__main__":
    G = nx.Graph()
    params = [(4, 1, 4), (16, 4, 4)]
    probs = [0, 8e-1]
    G, tree = newHSBM.generate_HSBM(params, probs)
    print(G.edges)
    id_generator = NewIDPartitionTreeNode(len(G.nodes))
    root = LinkagePlusPlus(G, k=4)
    Q: queue.Queue(PartitionTreeNode) = queue.Queue()
    Q.put(root)
    while not Q.empty():
        p = Q.get()
        print(p)
        if p.children is not None:
            for ch in p.children:
                Q.put(ch)
