import networkx as nx
import numpy as np

def canLabelFromGraph(graphEdges, types):

    """
    Compute the canonical label from the graph representation using the Weisfeiler-Lehman Algorithm.

    Parameters:
    graph_edges (list): List of edges in the graph.
    types (numpy.ndarray): Array of node types.

    Returns:
    str: The canonical label of the graph.
    """

    G = nx.Graph([list(row) for row in graphEdges])

    # coloring
    if types is not None:
        nx.set_node_attributes(G, np.array(types), "types")
        L = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(G, node_attr = "types")
    else:
        L = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(G)

    # Compute the canonical label using the Weisfeiler-Lehman Algorithm.
    return L


def nonCanLabelFromGraph(graphEdges, defectIndices):

    G = nx.Graph([list(row) for row in graphEdges])

    # coloring
    nx.set_node_attributes(G, np.array(defectIndices), "IDs")

    # Compute the canonical label using the Weisfeiler-Lehman Algorithm.
    L = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(G, node_attr = "IDs")
    
    return L