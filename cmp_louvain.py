from community import community_louvain
import community
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import vis_tree


def trans(partition):
    community2node = {}
    for k, v in partition.items():
        if v not in community2node:
            community2node[v] = set()
        community2node[v].add(k)
    return community2node


def LouvainLayers(G: nx.Graph):
    dendo = community_louvain.generate_dendrogram(G)
    # print(len(dendo))
    if len(dendo) > 1:
        res = []
        for level in range(len(dendo)):
            level_partition = trans(dendo[level])
            res.append(level_partition)

    res.append({'root': {k for k in res[-1].keys()}})
    res.reverse()
    return res



if __name__ == '__main__':
    G = nx.karate_club_graph()
    print(Louvain(G))
