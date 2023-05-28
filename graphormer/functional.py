from typing import Tuple, Dict, List

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from joblib import Parallel, delayed


def floyd_warshall_source_to_all(G, source, cutoff=None):
    if source not in G:
        raise nx.NodeNotFound("Source {} not in G".format(source))

    edges = {edge: i for i, edge in enumerate(G.edges())}

    level = 0  # the current level
    nextlevel = {source: 1}  # list of nodes to check at next level
    node_paths = {source: [source]}  # paths dictionary  (paths to key from source)
    edge_paths = {source: []}

    while nextlevel:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in G[v]:
                if w not in node_paths:
                    node_paths[w] = node_paths[v] + [w]
                    edge_paths[w] = edge_paths[v] + [edges[tuple(node_paths[w][-2:])]]
                    nextlevel[w] = 1

        level = level + 1

        if (cutoff is not None and cutoff <= level):
            break

    return node_paths, edge_paths


def all_pairs_shortest_path(G) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    paths = Parallel(n_jobs=2)(delayed(floyd_warshall_source_to_all)(G, n) for n in G)
    node_paths = {n: paths[n][0] for n in range(len(paths))}
    edge_paths = {n: paths[n][1] for n in range(len(paths))}
    return node_paths, edge_paths


def shortest_path_distance(data: Data):
    G = to_networkx(data)
    node_paths, edge_paths = all_pairs_shortest_path(G)
    return node_paths, edge_paths