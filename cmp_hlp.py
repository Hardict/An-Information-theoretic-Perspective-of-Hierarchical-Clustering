import networkx as nx
import copy
import random
import newHSBM


def LabelPropagation(G: nx.Graph):
    class_dict = dict()
    for nd in G.nodes:
        class_dict[nd] = nd
    flag = True
    # label propagation
    max_cycle = 50
    lis = list(G.nodes)
    flag = True
    for _ in range(max_cycle):
        previous_dict = copy.deepcopy(class_dict)
        random.shuffle(lis)
        for u in lis:
            dic = dict()
            max_relation = 0
            for v in G.adj[u]:
                w = 1
                if 'weight' in G.adj[u][v]:
                    w = G.adj[u][v]['weight']
                if class_dict[v] not in dic:
                    dic[class_dict[v]] = 0
                dic[class_dict[v]] += w
                if dic[class_dict[v]] > max_relation:
                    max_relation = dic[class_dict[v]]
                    class_dict[u] = class_dict[v]
        flag = False
        for key, val in previous_dict.items():
            if previous_dict[key] != class_dict[key]:
                flag = True
                break
        if flag == False:
            break
    return class_dict


def hypergraph(G: nx.Graph, node2community) -> nx.Graph:
    new_G = nx.Graph()
    for u, v, data in G.edges(data=True):
        if u not in node2community or v not in node2community:
            continue
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
        # hlp不应该考虑自环
        else:
            if new_G.has_edge(c_u, c_v):
                new_G[c_u][c_v]['weight'] += 2 * G[u][v]['weight']
            else:
                new_G.add_edge(c_u, c_v, weight=2 * G[u][v]['weight'])
    return new_G


def HLPLayers(G: nx.Graph):
    res = []
    H = copy.deepcopy(G)
    # print(nx.is_connected(G))
    max_cycle = len(G.nodes)

    cnt_class = len(G.nodes)
    while len(H.nodes) >= 2:
        max_cycle -= 1
        if max_cycle <= 0:
            break
        class_dict = LabelPropagation(H)
        dic = dict()
        for key, val in class_dict.items():
            if val not in dic:
                dic[val] = set()
            dic[val].add(key)
        # 如果G不连通需要将循环额外添加社区数不再改变的终止条件
        if cnt_class == len(dic.keys()):
            # print(1)
            new_dic = dict()
            new_dic['root'] = set()
            for key in dic.keys():
                new_dic['root'].add(key)
            res.append(new_dic)
            break
        cnt_class = len(dic.keys())
        res.append(dic)
        # print(nx.is_connected(H))
        H = hypergraph(H, class_dict)
    res.reverse()
    # print(res)
    # print(1)
    key = list(res[0].keys())[0]
    res[0]['root'] = res[0].pop(key)
    return res


import vis_tree

if __name__ == '__main__':
    # G = nx.karate_club_graph()
    # params = [(10, 1, 5), (100, 10, 5), (1000, 100, 5)]
    # probs = [0.00015, 0.065, 0.9]
    # probs = [2.5e-4, 4.5e-2, 0.9]
    params = [(5, 1, 3), (25, 5, 3), (250, 25, 5), (2500, 250, 5)]
    # params = [(5, 1, 3), (25, 5, 3), (125, 25, 3), (625, 125, 3)]
    # probs = [6e-6, 1.5e-3, 4.5e-2, 9e-1]
    # probs = [4e-6, 1.5e-3, 5.5e-2, 9e-1]
    probs = [2.5e-6, 4.5e-3, 6.5e-2, 9e-1]
    noweighted_G, tree = newHSBM.generate_HSBM(params, probs)
    # while nx.is_connected(noweighted_G) == False:
    #     noweighted_G, tree = newHSBM.generate_HSBM(params, probs)

    # noweighted_G: nx.Graph = nx.read_edgelist("graph.in", nodetype=int, data=False)
    print(noweighted_G)
    G = nx.Graph()
    G.add_weighted_edges_from([(u, v, 1) for u, v in noweighted_G.edges()])
    # print(G.adj[1])
    # print(LabelPropagation(G))
    layers = HLPLayers(G)
    for layer in layers:
        print(layer)
    root = vis_tree.build(layers)
    vis_tree.dfs(root, G)
    hlp_layer_class = dict()
    for k in range(len(layers)):
        # k 是第k层，根结点为第0层
        dic = dict()
        clss = vis_tree.get_k_layer_ground_truth(root, k)
        print(clss)
        for key in clss.keys():
            for nd in clss[key]:
                dic[nd] = key
        print(len(dic.keys()), dic)
