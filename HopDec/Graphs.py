import networkx as nx
import numpy as np

def graphLabel(graphEdges, canonical = 1, types = None, indices = None):


    G = nx.Graph([list(row) for row in graphEdges])

    if not canonical:
        if indices is None:
            ValueError('Non-canonical labeling requires indices to be set.')

        nx.set_node_attributes(G, np.array(indices), "IDs")
        L = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(G, node_attr = "IDs")

    else:
        if types is None:
            L = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(G)

        else:
            nx.set_node_attributes(G, np.array(types), "types")
            L = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(G, node_attr = "types")

    return L


def nDefectVolumes(graphEdges):

    def dfs(graph, start, visited):
        visited[start] = True
        for neighbor in graph[start]:
            if not visited[neighbor]:
                dfs(graph, neighbor, visited)

    def check_reachability(graph):
        num_nodes = len(graph)
        for start_node in range(num_nodes):
            visited = [False] * num_nodes
            dfs(graph, start_node, visited)
            if not all(visited):
                return False
        return True

    num_nodes = max(max(edge) for edge in graphEdges) + 1
    graph = [[] for _ in range(num_nodes)]

    for edge in graphEdges:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    if check_reachability(graph):
        return 1
    else:
        return 2