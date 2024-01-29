import queue
import networkx as nx
import graph_methods as graph
import numpy as np
import pickle


class TreeNode:
    def __init__(self, value=None, depth=None):
        self.value = value
        self.children = []
        self.depth = depth
        self.node_set = set()
        self.parent = None
        self.volume = 0
        self.g_val = 0
        self.cut_val = 0

    def add_child(self, node):
        self.children.append(node)
        node.parent = self

    def __str__(self):
        return self.value


def build(layer_list):
    q = queue.Queue()
    root = TreeNode('root', depth=0)
    q.put(root)
    for layer in layer_list:
        # print(layer)
        length = q.qsize()
        for _ in range(length):
            parent = q.get()
            # print(layer, parent.value)
            if parent.value in layer.keys():
                for val in layer[parent.value]:
                    child = TreeNode(val, depth=parent.depth + 1)
                    q.put(child)
                    parent.add_child(child)
    return root


def get_k_layer_ground_truth(node, k):
    res = dict()
    if node.depth > k:
        return res
    elif node.depth == k:
        res[node.value] = node.node_set

    for child in node.children:
        res.update(get_k_layer_ground_truth(child, k))
    return res


def calc_node_cut_val(node, G):
    for i in range(len(node.children)):
        for j in range(i + 1, len(node.children)):
            node.cut_val += graph.cut_value(G, node.children[i].node_set, node.children[j].node_set)


def dfs(node, G):
    if not node.children:
        node.node_set = {node.value}
        node.volume = sum(edge_data['weight'] for _, _, edge_data in G.edges(node.value, data=True))
        node.g_val = node.volume
    else:
        for child in node.children:
            dfs(child, G)
            node.node_set.update(child.node_set)
            node.volume += child.volume
            node.g_val += child.g_val
        calc_node_cut_val(node, G)
        node.g_val -= 2 * node.cut_val


def print_tree(node, depth, f):
    if node is not None:
        f.write('-' * depth + str(node.value) + ' ' + str(len(node.children)) + '\n')
        if depth > 100:
            return
        for child in node.children:
            print_tree(child, depth + 1, f)


def calc_SE(node, G_volume):
    res = 0
    if node.parent is not None:
        res = - node.g_val / G_volume * np.log2(node.volume / node.parent.volume)
    for child in node.children:
        res += calc_SE(child, G_volume)
    return res


def calc_HME(node, G_volume):
    res = 0
    if len(node.children) > 0 and node.g_val > 0:
        res -= node.g_val / G_volume * np.log2(node.g_val / (2 * node.cut_val + 2 * node.g_val))
    for child in node.children:
        if child.g_val > 0:
            res -= child.g_val / G_volume * np.log2(child.g_val / (2 * node.cut_val + 2 * node.g_val))
        res += calc_HME(child, G_volume)
    return res


def calc_Das(node):
    res = 0
    if node.children is not None:
        res = node.cut_val * len(node.node_set)
    for child in node.children:
        res += calc_Das(child)
    return res


def get_leaf_depth(node, depth):
    res = []
    if node.children is None or len(node.children) == 0:
        res = [depth]
        return res
    for child in node.children:
        if len(node.children) == 1:
            res += get_leaf_depth(child, depth)
        else:
            res += get_leaf_depth(child, depth + 1)
    return res


def calc_depth_balance_factor(depths):
    return np.std(depths)


def calc_size_balance_factor(node):
    if node.children is None or len(node.children) == 0:
        factor = 0
    elif len(node.children) == 2:
        left_child, right_child = node.children
        factor = len(node.node_set) * np.abs(len(left_child.node_set) - len(right_child.node_set)) / len(node.node_set)
        factor += calc_size_balance_factor(left_child)
        factor += calc_size_balance_factor(right_child)
    else:
        raise Exception(f'Not a binary tree')
    return factor


def calc_volume_balance_factor(node):
    if node.children is None or len(node.children) == 0:
        factor = 0
    elif len(node.children) == 2:
        left_child, right_child = node.children
        factor = node.volume * np.abs((left_child.volume - right_child.volume) / node.volume)
        factor += calc_volume_balance_factor(left_child)
        factor += calc_volume_balance_factor(right_child)
    else:
        raise Exception(f'Not a binary tree')
    return factor


def get_internal_nodes_volume_sum(node):
    if node.children is None or len(node.children) == 0:
        volume_sum = 0
    elif len(node.children) == 2:
        left_child, right_child = node.children
        volume_sum = node.volume
        volume_sum += get_internal_nodes_volume_sum(left_child)
        volume_sum += get_internal_nodes_volume_sum(right_child)
    else:
        raise Exception(f'Not a binary tree')
    return volume_sum


def get_internal_nodes_size_sum(node):
    if node.children is None or len(node.children) == 0:
        volume_sum = 0
    elif len(node.children) == 2:
        left_child, right_child = node.children
        volume_sum = len(node.node_set)
        volume_sum += get_internal_nodes_size_sum(left_child)
        volume_sum += get_internal_nodes_size_sum(right_child)
    else:
        raise Exception(f'Not a binary tree')
    return volume_sum


if __name__ == '__main__':
    print(1)