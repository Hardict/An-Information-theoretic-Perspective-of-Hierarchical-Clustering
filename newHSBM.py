import numpy as np
import networkx as nx


class TreeNode:
    def __init__(self, label=None, prob=None):
        self.label = label
        self.children = []
        self.parent = None
        self.prob = prob
        self.depth = 0
        self.node_set = set()
        self.depth = None

    def add_child(self, node):
        self.children.append(node)
        node.parent = self


class MultiTree:
    def __init__(self):
        self.root = TreeNode()
        self.label2leaf_node = {}
        self.count = 0

    def add_leaf(self, label, node):
        self.label2leaf_node[label] = node

    def find_node(self, label):
        if label not in self.label2leaf_node:
            return None
        return self.label2leaf_node[label]

    def lca_leaves(self, label1, label2):
        node1 = self.find_node(label1)
        node2 = self.find_node(label2)
        if node1 is None or node2 is None:
            return None
        visited = set()
        while node1 is not None or node2 is not None:
            if node1 is not None:
                if node1 in visited:
                    return node1
                visited.add(node1)
                node1 = node1.parent

            if node2 is not None:
                if node2 in visited:
                    return node2
                visited.add(node2)
                node2 = node2.parent
        return None

    # def build_tree(self, fanout, prob):
    #     self.build_tree_helper(self.root, fanout, depth = len(prob) - 1, prob = prob)
    #
    # def build_tree_helper(self, node, fanout, depth, prob):
    #     node.prob = prob[depth]
    #     if node is not None and depth > 0:
    #         for i in range(fanout[depth]):
    #             child = TreeNode()
    #             node.add_child(child)
    #             self.build_tree_helper(child, fanout, depth - 1, prob)

    def calc_fanout_layer(self, num_balls, num_boxes, min_balls):
        num_balls -= num_boxes * min_balls
        if num_balls < 0:
            raise Exception(f'num_balls < 0')
        boxes = [min_balls] * num_boxes
        for i in range(num_balls):
            box_index = random.randint(0, num_boxes - 1)
            boxes[box_index] += 1
        return boxes

    def calc_fanout(self, params):
        fanout = []
        for num_balls, num_boxes, min_balls in params:
            fanout.append(self.calc_fanout_layer(num_balls, num_boxes, min_balls))
        return fanout

    def build_tree(self, fanout, probs):
        if not fanout:
            return None
        queue = [self.root]
        level = 0
        id = 0
        while queue:
            # print('level', level)
            # print(fanout[level])
            for i in range(len(fanout[level])):
                node = queue.pop(0)
                node.prob = probs[level]
                # print(probs[level])
                if level < len(fanout) - 1:
                    for _ in range(fanout[level][i]):
                        child = TreeNode()
                        node.add_child(child)
                        queue.append(child)
                else:
                    # print(set(range(id, id + fanout[level][i])))
                    node.node_set = set(str(x) for x in range(id, id + fanout[level][i]))
                    id += fanout[level][i]
            level += 1

    def traverse_nodes(self):
        self.traverse_nodes_helper(self.root, self, 0)

    def traverse_nodes_helper(self, node, tree, depth):
        if node is None:
            return
        node.depth = depth
        for child in node.children:
            self.traverse_nodes_helper(child, tree, depth + 1)
            node.node_set.update(child.node_set)

        if not node.children:
            node.label = self.count
            self.count += 1
            tree.add_leaf(node.label, node)
        else:
            node.label = min([child.label for child in node.children])

    def get_k_layer_ground_truth(self, node, k):
        res = dict()
        if node.depth > k:
            return res
        elif node.depth == k:
            res[node.label] = node.node_set

        for child in node.children:
            res.update(self.get_k_layer_ground_truth(child, k))

        return res


# class TreeNode:
#     def __init__(self, val=None, children=None):
#         self.val = val
#         self.children = children or []
#         self.depth = 0  # 初始化深度为0
#
# def dfs(node, depth):
#     node.depth = depth  # 将当前节点的深度设置为传入的深度值
#     for child in node.children:
#         dfs(child, depth + 1)  # 递归遍历子节点，并将深度值+1
#
# # 示例使用
# root = TreeNode(1, [TreeNode(2), TreeNode(3, [TreeNode(4), TreeNode(5)]), TreeNode(6)])
# dfs(root, 0)

# 旧版实现
# def generate_HSBM(sizes, fanout, probs):
#     """
#     The method generates a networkx random graph according to the Hierarchical Stochastic Block Model (HSBM).
#     Formally, the graph consists of len(sizes) clusters on different of various sizes.
#     """
#     tree = MultiTree()
#     # tree.build_tree(fanout=fanout, prob=probs)
#     tree.traverse_leaf_nodes()
#     num = len(tree.label2leaf_node)
#     prob_matrix = np.zeros((num, num))
#     for i in range(num):
#         for j in range(num):
#             prob_matrix[i][j] = tree.lca_leaves(i, j).prob
#
#     G = nx.stochastic_block_model(sizes, prob_matrix)
#     for edge in G.edges():
#         G[edge[0]][edge[1]]['weight'] = 1.0
#     return G


def generate_HSBM(params, probs):
    """
    The method generates a networkx random graph according to the Hierarchical Stochastic Block Model (HSBM).
    Formally, the graph consists of len(sizes) clusters on different of various sizes.
    """
    tree = MultiTree()

    # tree.build_tree(fanout=fanout, prob=probs)
    fanout = tree.calc_fanout(params)
    # print(fanout)
    tree.build_tree(fanout=fanout, probs=probs)

    tree.traverse_nodes()
    # print(tree.root.label)
    # print(tree.root.node_set, 'd')
    for k in range(len(probs)):
        # k 是第k层，根结点为第0层
        res = tree.get_k_layer_ground_truth(tree.root, k)

    # print(tree.root.node_set)
    num = len(tree.label2leaf_node)
    prob_matrix = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            prob_matrix[i][j] = tree.lca_leaves(i, j).prob
    # print(prob_matrix.sum())
    sizes = fanout[-1]
    # print(sizes)
    # print(prob_matrix)
    G = nx.stochastic_block_model(sizes, prob_matrix)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1.0
    return G, tree


# def generate_HSBM_level_order(sizes, fanout, probs):
#     """
#     The method generates a networkx random graph according to the Hierarchical Stochastic Block Model (HSBM).
#     Formally, the graph consists of len(sizes) clusters on different of various sizes.
#     """

import random


def calc_fanout(num_balls, num_boxes, min_balls):
    num_balls -= num_boxes * min_balls
    if num_balls < 0:
        raise Exception(f'num_balls < 0')
    boxes = [min_balls] * num_boxes
    for i in range(num_balls):
        box_index = random.randint(0, num_boxes - 1)
        boxes[box_index] += 1
    return boxes


# # (num_balls, num_boxes, min_balls)
# params = [(5, 1, 5), (25, 5, 5), (250, 25, 10), (2500, 250, 10)]
# # [1000, 100, 10, 1], [0, 5, 5, 5], [0.8, 0.2, 0.015]
# probs = [6e-5, 1.5e-3, 4.5e-2, 0.9]

# params = [(5, 1, 3), (25, 5, 3), (125, 25, 3)]
# probs = [0.01, 0.1, 0.8]
#
# G, tree = generate_HSBM(params, probs)
# # tree.build_tree(fanout=fanout, prob=probs)
# # print(G)
# with open('edges.txt', 'w') as f:
#     for edge in G.edges:
#         f.write(f"{edge[0]} {edge[1]}\n")
# example
# sizes = np.full(35, 10)
# probs = [0.8, 0.2, 0.015]
# fanout = [0, 7, 5]
# G = generate_HSBM(sizes, fanout, probs)

# for edge in G.edges():
#     G[edge[0]][edge[1]]['weight'] = 1.0
#
# with open('edges.txt', 'w') as f:
#     for edge in G.edges:
#         f.write(f"{edge[0]} {edge[1]}\n")
# print(root.label, root.parent)
# for child in root.children:
#     print(child.label)
# node = TreeNode('0')
# build_tree(node, 3, 3)
# node1 = TreeNode(1)
# node2 = TreeNode(2)
# node1.add_child(node2)
# print(node1.label, node1.children[0].label, node1.parent)
