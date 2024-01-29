import networkx as nx
import scipy as sp
import numpy as np
def get_smallest_eigs(G, k):
    """
    Given a parameter k, we compute and return the k smallest eigenvalues
    of the normalised Laplacian matrix of G

    :param G: A networkx graph
    :param k: A number of desired eigenvalues
    :return: A sorted list of the k smallest eigenvalues of the normalised Laplacian of G
    """

    # Corner case if G is empty all eigenvalues are 0.
    if nx.is_empty(G):
        return [0] * k
    # Corner case k should be smaller than the number of vertices in G.
    elif k > G.number_of_nodes():
        raise Exception(f'{k} eigenvalues requested, but G has only {G.number_of_nodes()} nodes')

    normalized_laplacian_matrix = nx.normalized_laplacian_matrix(G, weight='weight')
    # print(normalized_laplacian_matrix)
    eigs = sp.sparse.linalg.eigsh(normalized_laplacian_matrix, k=k, which='SM', return_eigenvectors=False)
    # print('sp', eigs)
    # e = np.linalg.eigvals(normalized_laplacian_matrix.toarray())
    # e.sort()
    # print('np', e[:10])
    # e.sort()
    # print('np', e[:10])
    eigs.sort()

    return eigs

def volume(G, S):
    """
    Given a set of vertices S in G, we compute and return its volume in G,
    i.e. the sum of the degrees of vertices in S.

    :param G: A networkx graph
    :param S: A list of vertices in G
    :return: The volume of S in G
    """

    if nx.is_empty(G) or len(S) == 0:
        return 0
    return nx.volume(G, S, weight='weight')

# 必须保证S与T之间无交集
def cut_value(G, S, T):
    """
    Given two sets of vertices S and T, we compute and return the cut value between S and T,
    i.e. the sum of the weights of edges with one endpoint in S and the other in T,
    sometimes denoted as w(S, T).

    :param G: A networkx graph
    :param S: A list of vertices in G
    :param T: A list of vertices in G
    :return: The cut value between S and T in G
    """

    # Deal with the corner cases
    if nx.is_empty(G) or len(S) == 0 or len(T) == 0:
        return 0

    # cut_val stores the overall cut value
    cut_val = 0

    # If S and T have small sizes, we compute the cut by looping through S and T
    if G.number_of_nodes() ** 1.5 > len(S) * len(T):
        for u in S:
            for v in T:
                if G.has_edge(u, v):
                    if 'weight' in G[u][v]:
                        cut_val += G[u][v]['weight']
                    else:
                        cut_val += 1.0
    # If S and T has large size, we compute the cut by looping through the edges in G
    else:
        for u, v in list(G.edges()):
            if (u in set(S) and v in set(T)) or (v in set(S) and u in set(T)):
                if 'weight' in G[u][v]:
                    cut_val += G[u][v]['weight']
                else:
                    cut_val += 1.0
    return cut_val

def conductance(G, S):
    """
    Given a set of vertices S in G, we compute and return its conductance in G.
    The conductance of S in G is defined as the ratio w(S, Sc) / vol(S), where
    w(S, Sc) is the cut value between S and its complement Sc,
    and vol(S) is the volume of S in G

    :param G: A networkx graph
    :param S: A list of vertices in G
    :return: The conductance of S in G
    """

    vol = volume(G, S)

    # Corner cases if S is empty or has zero volume, by convention the conductance is 1.
    if len(S) == 0 or vol == 0:
        return 1

    S_complement_in_G = diff(G, S)
    return cut_value(G, S, S_complement_in_G) / vol

def varphi_conductance(G, S, P):
    """
    This method computes the relative conductance varphi(S, P) for a subset of nodes S of P,
    defined as the ratio (weight(S, P - S)) / (coeff * w(S, V - P)), where coeff = vol(P - S) / vol(P).

    :param G: A networkx graph
    :param S: A subset of nodes included in P
    :param P: An arbitrary subset of nodes in G
    :return: The relative conductance of S with respect to P
    """

    # The relative volume coefficient
    coeff = volume(G, diff(P, S)) / volume(G, P)

    S_complement_in_P = diff(P, S)
    P_complement_in_G = diff(G, P)

    if coeff == 0 or cut_value(G, S, P_complement_in_G) == 0:
        return 1
    return cut_value(G, S, S_complement_in_P) / (coeff * cut_value(G, S, P_complement_in_G))

def diff(A, B):
    return set(A) - set(B)






