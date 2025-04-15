from scipy.spatial import KDTree
import numpy as np
from collections import defaultdict

#DFS Helpers
def build_graph(skeleton):
    """
    Build a graph from the skeleton. Each pixel in the skeleton is a node.
    Nodes are connected to their 8-connected neighbors.
    """
    rows, cols = np.where(skeleton > 0)
    nodes = list(zip(rows, cols))
    tree = KDTree(nodes)

    graph = defaultdict(list)
    for i, node in enumerate(nodes):
        # Find neighbors within distance sqrt(2) (diagonal distance in 8-connected grid)
        distances, indices = tree.query(node, k=len(nodes), distance_upper_bound=np.sqrt(2) + 1e-5)
        for j, dist in zip(indices, distances):
            if dist < np.sqrt(2) + 1e-5 and j != i:  # Avoid self-loops
                graph[node].append(nodes[j])

    return graph


def dfs_longest_path(graph, start):
    """
    Perform a DFS to find the longest path in the graph.
    """
    visited = set()
    stack = [(start, [start])]
    longest_path = []

    while stack:
        current_node, path = stack.pop()
        if current_node in visited:
            continue

        visited.add(current_node)
        if len(path) > len(longest_path):
            longest_path = path

        for neighbor in graph[current_node]:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))

    return longest_path