from typing import Union

from torch import nn
from torch_geometric.data import Data

from graphormer.functional import shortest_path_distance
from graphormer.layers import GraphormerEncoderLayer, CentralityEncoding, SpatialEncoding


class Graphormer(nn.Module):
    def __init__(self, num_layers, input_node_dim, node_dim, input_edge_dim, edge_dim, output_dim, n_heads,
                 max_in_degree, max_out_degree, max_path_distance):
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
            d_model=self.node_dim
        )

        self.spatial_encoding = SpatialEncoding(
            max_path_distance=max_path_distance,
        )

        self.layers = nn.ModuleList([
            GraphormerEncoderLayer(node_dim=self.node_dim, edge_dim=self.edge_dim, n_heads=self.n_heads) for _ in
            range(self.num_layers)
        ])

        self.node_out_lin = nn.Linear(self.node_dim, self.output_dim)

    def forward(self, data: Union[Data]):
        """
        :param data:
        :return:
        """
        x = data.x.float()
        edge_index = data.edge_index.long()
        edge_attr = data.edge_attr.float()
        ptr = None if type(data) == Data else data.ptr

        x = self.node_in_lin(x)
        edge_attr = self.edge_in_lin(edge_attr)

        x = self.centrality_encoding(x, edge_index)
        node_paths, edge_paths = shortest_path_distance(data)
        b = self.spatial_encoding(x, node_paths)

        for layer in self.layers:
            x = layer(x, edge_attr, b, edge_paths, ptr)

        x = self.node_out_lin(x)

        return x
