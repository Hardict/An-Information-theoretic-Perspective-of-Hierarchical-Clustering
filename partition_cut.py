import networkx as nx
import numpy as np
import cheeger_cut
import graph_methods as graph
import logging


def compute_improved_partition(G, k):
    # Initialise the parameters for the algorithm
    # We compute the smallest k eigenvalues of the normalized Laplacian matrix of G
    # 此处 k += 1的目的是因为GT算法中 1 <= l < k
    k += 1
    eigs = graph.get_smallest_eigs(G, k)
    logging.info("The array contains the following values: %s", eigs)
    # We assign the (k - 1)'th and k'th eigenvalues to the corresponding variables
    lambda_k = eigs[k - 1]
    lambda_k_minus_1 = eigs[k - 2]
    c_0 = 1
    # print(lambda_k / 10, 30 * c_0 * (k ** 5) * np.sqrt(lambda_k_minus_1))
    rho_star = min(lambda_k / 10, 30 * c_0 * (k ** 5) * np.sqrt(lambda_k_minus_1))
    logging.info('rho_star : %s', rho_star)
    phi_in = lambda_k / (140 * (k ** 2))
    logging.info(' phi_in : %s', phi_in)
    phi_in = max(phi_in, 2 * lambda_k_minus_1)
    logging.info(' phi_in : %s', phi_in)
    rho_star = max(phi_in, rho_star)
    logging.info('rho_star : %s', rho_star)
    clusters = get_clusters_and_critical_nodes(G, k, rho_star, phi_in)

    return clusters


def get_clusters_and_critical_nodes(G, k, rho_star, phi_in):
    """
        The implementation of the main body of the partitioning Algorithm.
        The main while-loop of the algorithm is executed as long as a refinement is still possible.

        :param phi_in: An algorithm parameter used to lower bound the inner conductance of each cluster
        :param rho_star: A technical parameter of the algorithm
        :param G: A networkx graph
        :param k: The (supposed) number of clusters

        :return: a list containing an l-wise partitioning of the nodes of G, for some l <= k
        """
    # A list of vertices in the graph G
    vertices = list(G.nodes())
    # Initially the graph contains one cluster P_1 = V with core set core_P_1 = P_1.
    core_P_1 = vertices[:]
    P_1 = core_P_1[:]
    # num_clusters is the variable denoting the current number of clusters
    num_clusters = 1
    # clusters is a list storing the current cluster structure of G (i.e. P_1, ..., P_l)
    clusters = [P_1]
    # core_sets is a list containing the current core_subsets of each cluster.
    # (i.e. core_P_1, ..., core_P_(num_clusters) with core_P_i being a subset of P_i)
    core_sets = [core_P_1]
    # The main loop of the algorithm. We continue as long as an update is possible

    GT_update_is_found = True
    iteration = 0
    while GT_update_is_found:
        iteration += 1
        logging.info('iteration : %s', iteration)
        logging.info('num_clusters : %s', num_clusters)
        # First we check if a GT_update is possible
        GT_update_is_found, index_cluster_to_update = check_if_GT_update_is_found(G, clusters, core_sets, phi_in)

        if GT_update_is_found:
            # GT_update_is_done是为了实现if间的相互排斥
            GT_update_is_done = False
            # Notation of the corresponding sets of vertices
            P_i = clusters[index_cluster_to_update]
            core_P_i = core_sets[index_cluster_to_update]

            S = cheeger_cut.cheeger_cut(G.subgraph(P_i))

            core_P_i_intersect_S = intersect(core_P_i, S)
            core_P_i_minus_S = diff(core_P_i, S)
            S_minus_core_P_i = diff(S, core_P_i)
            core_P_i_union_S = union(core_P_i, S)
            core_P_i_union_S_complement_in_P = diff(P_i, core_P_i_union_S)
            P_i_minus_core_P_i = diff(P_i, core_P_i)

            # Without loss of generality we assume vol(core_P_i_intersect_S) <= vol(core_P_i) / 2
            if vol(G, core_P_i_intersect_S) > vol(G, core_P_i_minus_S):
                core_P_i_intersect_S, core_P_i_minus_S = core_P_i_minus_S, core_P_i_intersect_S
                S_minus_core_P_i, core_P_i_union_S_complement_in_P = core_P_i_union_S_complement_in_P, S_minus_core_P_i

            # First "if" in the algorithm
            if not GT_update_is_done and is_first_if_condition_satisfied(G, core_P_i_intersect_S, core_P_i_minus_S, k,
                                                                         num_clusters, rho_star):
                make_new_cluster_with_core_P_i_minus_S(core_P_i_intersect_S, core_P_i_minus_S, clusters, core_sets,
                                                       index_cluster_to_update)
                num_clusters += 1
                # A sanity check update
                num_clusters = min(num_clusters, k)

                GT_update_is_done = True

            # Second "if" in the algorithm
            if not GT_update_is_done and is_second_if_condition_satisfied(G, core_P_i_intersect_S, core_P_i_minus_S,
                                                                          core_P_i, k):
                update_core_P_i_to_core_P_i_intersect_S_or_core_P_i_minus_S(G, core_P_i_intersect_S, core_P_i_minus_S,
                                                                            core_sets, index_cluster_to_update)
                print('second')
                GT_update_is_done = True

            # Third "if" in the algorithm
            if not GT_update_is_done and is_third_if_condition_satisfied(G, S_minus_core_P_i, k, num_clusters,
                                                                         rho_star):
                make_new_cluster_with_S_minus_core_P_i(S_minus_core_P_i, clusters, core_sets, index_cluster_to_update)

                num_clusters += 1
                # A sanity check update
                num_clusters = min(num_clusters, k)

                GT_update_is_done = True

            # Forth "if" in the algorithm
            if not GT_update_is_done and is_forth_if_condition_satisfied(G, P_i_minus_core_P_i, clusters,
                                                                         index_cluster_to_update):
                GT_update_is_done = True

            # Fifth "if" in the algorithm
            if not GT_update_is_done and is_fifth_if_condition_satisfied(G, core_P_i_union_S_complement_in_P, clusters,
                                                                         index_cluster_to_update):
                GT_update_is_done = True

            if not GT_update_is_done:
                raise Exception('No GT_update performed in iteration')

        for i, cluster in enumerate(clusters):
            logging.info('P[%s] %s : %s', i, len(cluster), cluster)

        for core_P_i in core_sets:
            logging.info('core_P[%s] %s : %s', i, len(core_P_i), core_P_i)
        if len(clusters) == k - 1:
            GT_update_is_found = False
    return clusters


def check_if_GT_update_is_found(G, clusters, core_sets, phi_in):
    """
    This method checks if the condition of the while-loop is satisfied,
    i.e. if a refinement of the current partition is possible

    :param G: A networkx graph
    :param clusters: A list of clusters corresponding to the current partitioning
    :param core_sets: A list of core_sets corresponding to each of the clusters
    :param phi_in: A parameter corresponding to the inner conductance

    :return: A pair (GT_update_is_found, index_of_cluster_to_update). The first
    element is assigned a boolean value and is True if the while condition is satisfied.
    The second element is the index i of the cluster to be updated
    """

    # Initialise the two returned values
    GT_update_is_found = False
    index_of_cluster_to_update = -1

    # Check if there exists a sparse cut cheeger cut. That is, for each cluster P_i with cheeger cut S check if
    # phi_G[P_i](S) < phi_in
    min_phi_in = phi_in
    for i in range(len(clusters)):
        H = G.subgraph(clusters[i])
        S = cheeger_cut.cheeger_cut(H)
        logging.info('%s : %s', len(H), H)
        logging.info('%s : %s', len(S), S)
        logging.info('graph.conductance(H, S) : %s', graph.conductance(H, S))
        logging.info('graph.conductance(H, diff(H, S)) : %s', graph.conductance(H, diff(H, S)))

        # print('phi_in', phi_in)
        # print('min_phi_in', min_phi_in)
        this_phi_in = max(graph.conductance(H, S), graph.conductance(H, diff(H, S)))
        # print('this_phi_in', this_phi_in)
        if this_phi_in < min_phi_in:
            GT_update_is_found, index_of_cluster_to_update = True, i
            min_phi_in = this_phi_in
            logging.info('success, phi_G[P_[%s]](S) < phi_in', i)
            # break
        else:
            logging.info('fail')
    # Check if we can refine the current partition. That is, check if there are i and j with
    # w(P_i - core(P_i) -> P_i) < w(P_i - core(P_i) -> P_j)
    if not GT_update_is_found:
        logging.info('entering w(P_i - core(P_i) -> P_i) < w(P_i - core(P_i) -> P_j)')
        for i in range(len(clusters)):
            P_i, core_P_i = clusters[i], core_sets[i]
            core_P_i_complement_in_P_i = diff(P_i, core_P_i)
            for j in range(len(clusters)):
                if j != i:
                    P_j = clusters[j]
                    if weight(G, core_P_i_complement_in_P_i, P_i) < weight(G, core_P_i_complement_in_P_i, P_j):
                        GT_update_is_found, index_of_cluster_to_update = True, i
                        logging.info('i, j : %s, %s', i, j)
                        break
            # Once an update is found we break the loop
            if GT_update_is_found:
                break
    return GT_update_is_found, index_of_cluster_to_update


def is_first_if_condition_satisfied(G, core_P_i_intersect_S, core_P_i_minus_S, k, num_clusters, rho_star):
    """
    This method checks if the first "if"-condition of the main algorithm is satisfied
    :param G: A networkx graph
    :param core_P_i_intersect_S: A list containing the vertices in the set core_P_i_intersect_S defined as core_P_i_intersect_S = intersect(core(P_i), S)
    :param core_P_i_minus_S: A list containing the vertices in the set core_P_i_minus_S defined as core_P_i_minus_S = core(P_i) - S
    :param k: The target number of clusters
    :param num_clusters: The current number of clusters in the algorithm. This is the parameter l in the paper
    :param rho_star: A parameter rho_star as defined in the paper and in the method compute_improved_partition()

    :return: True if the first "if"-condition of the algorithm is satisfied and False otherwise
    """
    phi_core_P_i_intersect_S = graph.conductance(G, core_P_i_intersect_S)
    phi_core_P_i_minus_S = graph.conductance(G, core_P_i_minus_S)
    # Compute the expression on the right hand side of the "if"-condition
    right_hand_side = (1 + 1.0 / k) ** (num_clusters + 1) * rho_star
    if max(phi_core_P_i_intersect_S, phi_core_P_i_minus_S) <= right_hand_side:
        return True
    else:
        return False


def is_second_if_condition_satisfied(G, core_P_i_intersect_S, core_P_i_minus_S, core_P_i, k):
    """
    This method checks if the second "if"-condition of the main algorithm is satisfied
    :param G: A networkx graph
    :param core_P_i_intersect_S: A list containing the vertices in the set core_P_i_intersect_S defined as core_P_i_intersect_S = intersect(core(P_i), S)
    :param core_P_i_minus_S: A list containing the vertices in the set core_P_i_minus_S defined as core_P_i_minus_S = core(P_i) - S
    :param core_P_i: The core set core(P_i) of cluster P_i
    :param k: The target number of clusters

    :return: True if the second "if"-condition of the algorithm is satisfied and False otherwise
    """
    varphi_core_P_i_intersect_S = graph.varphi_conductance(G, core_P_i_intersect_S, core_P_i)
    varphi_core_P_i_minus_S = graph.varphi_conductance(G, core_P_i_minus_S, core_P_i)
    if min(varphi_core_P_i_intersect_S, varphi_core_P_i_minus_S) <= 1.0 / (3 * k):
        return True
    else:
        return False


def is_third_if_condition_satisfied(G, S_minus_core_P_i, k, num_clusters, rho_star):
    """
    This method checks if the third "if"-condition of the main algorithm is satisfied
    :param G: A networkx graph
    :param S_minus_core_P_i: A list containing the vertices in the set S_minus_core_P_i defined as S_minus_core_P_i = S - core(P_i)
    :param k: The target number of clusters
    :param num_clusters: The current number of clusters in the algorithm. This is the parameter l in the paper
    :param rho_star: A parameter rho_star as defined in the paper and in the method compute_improved_partition()

    :return: True if the third "if"-condition of the algorithm is satisfied and False otherwise
    """
    phi_S_minus_core_P_i = graph.conductance(G, S_minus_core_P_i)
    # Compute the expression on the right hand side of the "if"-condition
    right_hand_side = ((1 + 1.0 / k) ** (num_clusters + 1)) * rho_star
    if phi_S_minus_core_P_i <= right_hand_side:
        return True
    else:
        return False


def is_forth_if_condition_satisfied(G, P_i_minus_core_P_i, clusters, index_cluster_to_update):
    # If there is a cluster P_j s.t. w(P_i - core_P_i -> P_i)  < w(P_i - core_P_i -> P_j),
    # then merge (P_i - core_P_i) with argmax_(P_j){w(P_i - core_P_i -> P_j)}

    # Find the index j of argmax_(P_j){w(P_i - core_P_i -> P_j)}.
    max_cluster_index = find_cluster_P_j_that_maximises_weight_from_T_to_P_j(G, P_i_minus_core_P_i, clusters)
    # Forth "if" in the algorithm.
    if max_cluster_index != index_cluster_to_update:
        move_subset_T_from_P_i_to_P_j(P_i_minus_core_P_i, clusters, index_cluster_to_update, max_cluster_index)
        return True
    else:
        return False


def is_fifth_if_condition_satisfied(G, core_P_i_union_S_complement_in_P, clusters, index_cluster_to_update):
    # If there is a cluster P_j s.t. w(S_minus -> P_i)  < w(S_minus -> P_j),
    # then merge S_minus with argmax_(P_j){w(S_minus -> P_j)}

    # Find the index j of argmax_(P_j){w(S_minus -> P_j)}.
    max_cluster_index = find_cluster_P_j_that_maximises_weight_from_T_to_P_j(G, core_P_i_union_S_complement_in_P,
                                                                             clusters)
    # Fifth "if" in the algorithm
    if max_cluster_index != index_cluster_to_update:
        move_subset_T_from_P_i_to_P_j(S_minus_core_P_i, clusters, index_cluster_to_update, max_cluster_index)
        return True
    else:
        return False


def find_cluster_P_j_that_maximises_weight_from_T_to_P_j(G, T, clusters):
    """
    Given a subset of vertices T, this method finds and returns the index j of cluster P_j that maximises w(T -> P_j)

    :param G: A networkx graph
    :param T: A list of vertices
    :param clusters: A list of clusters, stored as lists of vertices

    :return: The index j of cluster P_j that maximises w(T -> P_j)
    """
    max_cluster_index = 0
    max_weight = weight(G, T, clusters[0])
    for j in range(len(clusters)):
        P_j = clusters[j]
        weight_from_T_to_P_j = weight(G, T, P_j)
        if j != max_cluster_index and max_weight < weight_from_T_to_P_j:
            max_cluster_index = j
            max_weight = weight_from_T_to_P_j

    return max_cluster_index


def make_new_cluster_with_core_P_i_minus_S(core_P_i_intersect_S, core_P_i_minus_S, clusters, core_sets, i):
    """
    This method corresponds to the updates performed should the "if"-conditions #1 be satisfied.
    Given a partition core_P_i_intersect_S, core_P_i_minus_S of core(P_i), the method updates core(P_i) <- core_P_i_intersect_S and creates a new cluster with the
    vertices in core_P_i_minus_S

    :param core_P_i_intersect_S: A list of vertices such that core_P_i_intersect_S is a subset of core(P_i)
    :param core_P_i_minus_S: A list of vertices such that core_P_i_minus_S = core(P_i) - core_P_i_intersect_S
    :param clusters: A list of clusters, stored as lists of vertices
    :param core_sets: A list of core_sets, stored as lists of vertices. Each core_sets[i] is the core set of core(P_i)
    :param i: The index of the cluster P_i
    """
    # Update core_P_i = core_P_i_intersect_S
    core_sets[i] = core_P_i_intersect_S[:]
    # Update P_i = P_i - core_P_i_minus_S
    clusters[i] = diff(clusters[i], core_P_i_minus_S)
    # Update P_(num_clusters + 1) = core_P_(num_clusters + 1) = core_P_i_minus_S
    clusters.append(core_P_i_minus_S[:])
    core_sets.append(core_P_i_minus_S[:])


def update_core_P_i_to_core_P_i_intersect_S_or_core_P_i_minus_S(G, core_P_i_intersect_S, core_P_i_minus_S, core_sets,
                                                                i):
    """
    This method corresponds to the updates performed should the "if"-conditions #2 be satisfied.
    Given a partition core_P_i_intersect_S, core_P_i_minus_S of core(P_i), the method updates core(P_i) to either core_P_i_intersect_S or core_P_i_minus_S of lower conductance

    :param G: A networkx graph
    :param core_P_i_intersect_S: A list of vertices such that core_P_i_intersect_S is a subset of core(P_i)
    :param core_P_i_minus_S: A list of vertices such that core_P_i_minus_S = core(P_i) - core_P_i_intersect_S
    :param core_sets: A list of core_sets, stored as lists of vertices. Each core_sets[i] is the core set core(P_i)
    :param i: The index of the cluster P_i
    """
    # Update core_P_i to either core_P_i_intersect_S or core_P_i_minus_S of minimum conductance
    if graph.conductance(G, core_P_i_intersect_S) > graph.conductance(G, core_P_i_minus_S):
        core_P_i_minus_S, core_P_i_intersect_S = core_P_i_intersect_S, core_P_i_minus_S
    core_sets[i] = core_P_i_intersect_S[:]


def make_new_cluster_with_S_minus_core_P_i(S_minus_core_P_i, clusters, core_sets, i):
    """
    This method corresponds to the update performed should the "if"-condition #3 be satisfied.
    Given a subset S_minus_core_P_i of P_i, the method creates a new cluster with the vertices in S_minus_core_P_i

    :param S_minus_core_P_i: A list of vertices
    :param clusters: A list of clusters, stored as lists of vertices
    :param core_sets: A list of core_sets, stored as lists of vertices. Each core_sets[i] is the core set core(P_i)
    :param i: The index of the cluster P_i
    """
    # Update P_i = P_i - S_minus_core_P_i
    clusters[i] = diff(clusters[i], S_minus_core_P_i)

    # Update P_(num_clusters + 1) = core_P_(num_clusters + 1) = S_minus_core_P_i
    core_sets.append(S_minus_core_P_i[:])
    clusters.append(S_minus_core_P_i[:])


def move_subset_T_from_P_i_to_P_j(T, clusters, i, j):
    """
    This method corresponds to the updates performed should the "if"-conditions #4, #5 be satisfied.
    Given a subset T of P_i the method moves the set T to cluster P_j

    :param T: A list of vertices
    :param clusters: A list of clusters, stored as lists of vertices
    :param i: The index of the original cluster P_i
    :param j: The index of the target cluster P_j
    """
    # Update P_i = P_i - T
    clusters[i] = diff(clusters[i], T)

    # Merge T with P_j, where P_j is argmax_(P_j){w(T -> P_j)}
    clusters[j] = clusters[j] + T


# The intersect function that takes as input two lists (of vertices) A and B, and returns the list consisting of
# elements that are both in A and in B
def intersect(A, B):
    return list(set(A) & set(B))


def union(A, B):
    return list(set(A) | set(B))


# The diff function that takes as input two lists (of vertices) A and B, and returns the list consisting of
# elements in A that are not in B
def diff(A, B):
    return list(set(A) - set(B))


# The weight function w(S -> T) defined to be sum of the weights of the edges with one endpoint in S and the other in T / S
def weight(G, S, T):
    return graph.cut_value(G, S, diff(T, S))


def vol(G, S):
    return graph.volume(G, S)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)

    # example
    from newHSBM import generate_HSBM

    probs = [0.015, 0.2, 0.8]
    params = [(5, 1, 5), (35, 5, 7), (350, 35, 10)]
    noweighted_G, tree = generate_HSBM(params, probs)
    G = nx.Graph()
    G.add_weighted_edges_from([(str(u), str(v), 1) for u, v in noweighted_G.edges()])
    print(G)
    # clusters = compute_improved_partition(G, 5)
    # for cluster in clusters:
    #     print(len(cluster), cluster)

    import sys

    sys.path.append(
        r"hierarchical-clustering-well-clustered-graphs-main")
    from Tree_Construction import agg_constr
    import prune_merge

    # # example
    # G = nx.read_edgelist('karate_club.txt', nodetype = str, data = False)
    # # clusters = compute_improved_partition(G, 4)
    tree_al = agg_constr.build_agg_tree(G, 'average_linkage')
    # tree_sl = agg_constr.build_agg_tree(G, 'single_linkage')
    # tree_cl = agg_constr.build_agg_tree(G, 'complete_linkage')
    # # only 2 can work
    # tree_pm = prune_merge.prune_merge(G, 2)
    # # ours_clusters = compute_improved_partition(G, 3)
    # # for cluster in ours_clusters:
    # #     print(cluster)
    print(tree_al.print_layers())
    # print(tree_sl.print_layers())
    # print(tree_cl.print_layers())
    # print(tree_pm.print_layers())
    print(tree_al.get_tree_cost())
    # print(tree_sl.get_tree_cost())
    # print(tree_cl.get_tree_cost())
    # print(tree_pm.get_tree_cost())
    # # for cluster in clusters:
    # #     print(cluster, len(cluster))
