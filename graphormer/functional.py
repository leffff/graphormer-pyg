from typing import Tuple, Dict, List

import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from joblib import Parallel, delayed


def single_source_shortest_path(G, source, cutoff=None):
    if source not in G:
        raise nx.NodeNotFound("Source {} not in G".format(source))

    edges = list(G.edges())

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
                    edge_paths[w] = []
                    nextlevel[w] = 1

        level = level + 1

        if (cutoff is not None and cutoff <= level):
            break

    for v in node_paths:
        node_path = node_paths[v]
        for i in range(len(node_path[:-1])):
            index = edges.index((node_path[i], node_path[i + 1]))
            edge_paths[v].append(index)

    return node_paths, edge_paths


def fast_single_source_shortest_path(G, source, cutoff=None):
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
    paths = Parallel(n_jobs=12)((fast_single_source_shortest_path)(G, n) for n in G)
    node_paths = {n: paths[n][0] for n in range(len(paths))}
    edge_paths = {n: paths[n][1] for n in range(len(paths))}
    return node_paths, edge_paths


def shortest_path_distance(data: Data):
    G = to_networkx(data)
    node_paths, edge_paths = all_pairs_shortest_path(G)
    return node_paths, edge_paths