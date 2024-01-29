# Todo: 如何根据一棵树，树中每个中间结点代表cross该中间结点的边的概率，来随机生成一幅图
import networkx as nx

# sizes = [20, 10, 10]
# probs = [[0.05, 0.01, 0.01], [0.01, 0.21, 0.06], [0.01, 0.06, 0.24]]
# G = nx.stochastic_block_model(sizes, probs)
# for edge in G.edges():
#     G[edge[0]][edge[1]]['weight'] = 1.0
# G = generate_HSBM(sizes, probs)
#
# adjacency_list = {}
# for node in G.nodes():
#     neighbors = set(G.neighbors(node))
#     adjacency_list[node] = neighbors
#
# print(adjacency_list)

# calc_nmi
import numpy as np
from collections import Counter


def calc_nmi(cluster1, cluster2):
    # 计算两个聚类划分的NMI
    n = len(cluster1)
    assert len(cluster2) == n
    # 计算每个聚类划分的簇分布
    count1 = Counter(cluster1)
    count2 = Counter(cluster2)
    # 计算联合分布
    joint_count = Counter(zip(cluster1, cluster2))
    # 计算各自簇分布的概率
    p1 = {key: value / n for key, value in count1.items()}
    p2 = {key: value / n for key, value in count2.items()}
    # print(p1)
    # print(p2)
    # 计算联合分布的概率
    # print(joint_count)
    p_joint = {key: value / n for key, value in joint_count.items()}
    # print(p_joint)
    # 计算互信息
    mi = sum([p * np.log2(p / (p1[i] * p2[j])) for (i, j), p in p_joint.items()])
    # 计算标准互信息
    h1 = -sum([p * np.log2(p) for p in p1.values()])
    h2 = -sum([p * np.log2(p) for p in p2.values()])
    # print(h1, h2)
    nmi = mi / ((h1 + h2) / 2 + 1e-12)
    return nmi


def calc_jaccard(cluster1, cluster2):
    """
    :param cluster1: 划分1中每个点对应的类(从0开始)
    :param cluster2: 划分2中每个点对应的类(从0开始)
    :return:
    """
    jaccard = 0
    n = len(cluster1)
    assert len(cluster2) == n
    dict1 = dict()
    dict2 = dict()
    for i in range(n):
        if cluster1[i] not in dict1:
            dict1[cluster1[i]] = set()
        dict1[cluster1[i]].add(i)
        if cluster2[i] not in dict2:
            dict2[cluster2[i]] = set()
        dict2[cluster2[i]].add(i)
    for i in range(n):
        A: set = dict1[cluster1[i]]
        B: set = dict2[cluster2[i]]
        AB = A.intersection(B)
        jaccard += (len(AB)) / (len(A) + len(B) - len(AB))
    jaccard /= n
    return jaccard


if __name__ == '__main__':
    cluster1 = [1, 1, 2, 3, 3, 3, 3]
    cluster2 = [1, 1, 1, 3, 3, 3, 3]
    print(calc_jaccard(cluster1,cluster2))
