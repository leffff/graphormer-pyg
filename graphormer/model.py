from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data, Batch

from graphormer.functional import shortest_path_distance, batched_shortest_path_distance
from graphormer.layers import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding
from graphormer.utils import node_path_matrix_form_dict


class Graphormer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 input_node_dim: int,
                 node_dim: int,
                 input_edge_dim: int,
                 edge_dim: int,
                 output_dim: int,
                 n_heads: int,
                 max_in_degree: int,
                 max_out_degree: int,
                 max_path_distance: int):
        """
        :param num_layers: number of Graphormer layers
        :param input_node_dim: input dimension of node features
        :param node_dim: hidden dimensions of node features
        :param input_edge_dim: input dimension of edge features
        :param edge_dim: hidden dimensions of edge features
        :param output_dim: number of output node features
        :param n_heads: number of attention heads
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param max_path_distance: max pairwise distance between two nodes
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_node_dim = input_node_dim
        self.node_dim = node_dim
        self.input_edge_dim = input_edge_dim
        self.edge_dim = edge_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.max_path_distance = max_path_distance

        self.node_in_lin = nn.Linear(self.input_node_dim, self.node_dim)
        self.edge_in_lin = nn.Linear(self.input_edge_dim, self.edge_dim)

        self.centrality_encoding = CentralityEncoding(
            max_in_degree=self.max_in_degree,
            max_out_degree=self.max_out_degree,
            node_dim=self.node_dim
        )

        self.spatial_encoding = SpatialEncoding(
            max_path_distance=max_path_distance,
        )

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(node_dim=self.node_dim, edge_dim=self.edge_dim, n_heads=self.n_heads,
                                   max_path_distance=self.max_path_distance) for _ in
            range(self.num_layers)
        ])

        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

    def forward(self, data: Data | Batch) -> torch.Tensor:
        """
        :param data: input graph of batch of graphs
        :return: torch.Tensor, output node embeddings
        """
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()
        num_nodes = x.shape[0]

        if type(data) == Data:
            ptr = None
            node_paths, edge_paths = shortest_path_distance(data)
        else:
            ptr = data.ptr
            node_paths, edge_paths = batched_shortest_path_distance(data)

        print("PATHS CALCULATED")

        node_path_distance_matrix = node_path_matrix_form_dict(node_paths, num_nodes)
        print("PATHS CONVERTED")

        x = self.node_in_lin(x)
        print("NODE EMBEDDINGS")
        edge_attr = self.edge_in_lin(edge_attr)
        print("EDGE EMBEDDINGS")
        x = self.centrality_encoding(x, edge_index)
        print("CENTRALITY ENCODING")
        b = self.spatial_encoding(node_path_distance_matrix)
        print("SPATIAL ENCODING")

        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths, node_path_distance_matrix, ptr)
            print("LAYER")
        x = self.node_out_lin(x)
        print("OUT")
        return x
