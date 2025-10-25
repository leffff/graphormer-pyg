from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        """
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max in degree of nodes
        :param node_dim: hidden dimensions of node features
        """
        super().__init__()
        self.node_dim = node_dim
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    def forward(self, x: torch.Tensor, in_degree: torch.LongTensor, out_degree: torch.LongTensor) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param in_degree: in_degree of each node
        :param out_degree: out_degree of each node
        :return: torch.Tensor, node embeddings after Centrality encoding
        """

        x += self.z_in[in_degree] + self.z_out[out_degree]

        return x


class SpatialEncoding(nn.Module):
    def __init__(self, max_path_distance: int):
        """
        :param max_path_distance: max pairwise distance between nodes
        """
        super().__init__()
        self.max_path_distance = max_path_distance

        self.b = nn.Parameter(torch.randn(self.max_path_distance))

    def forward(self, x: torch.Tensor, node_paths_length: torch.Tensor) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param node_paths_length: tensor of shape [num_nodes, num_nodes] containing path lengths
        :return: torch.Tensor, spatial Encoding matrix
        """
        device = next(self.parameters()).device
        spatial_matrix = torch.zeros((x.shape[0], x.shape[0]), device=device)
        
        mask_of_nonexistent_paths = (node_paths_length == 0)
        indices = torch.clamp(node_paths_length-1, 0, len(self.b) - 1)
        spatial_matrix = F.embedding(indices, self.b.unsqueeze(1)).squeeze(-1)
        spatial_matrix[mask_of_nonexistent_paths] = 0

        return spatial_matrix


def dot_product(x1, x2) -> torch.Tensor:
    return (x1 * x2).sum(dim=1)


class EdgeEncoding(nn.Module):
    def __init__(self, edge_dim: int, max_path_distance: int):
        """
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_dim = edge_dim
        self.max_path_distance = max_path_distance
        self.edge_vector = nn.Parameter(torch.randn(self.max_path_distance, self.edge_dim))

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_paths_tensor, edge_paths_length) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param edge_paths_tensor: 3D tensor of shape [num_nodes, num_nodes, max_path_distance] containing edge indices
        :param edge_paths_length: 2D tensor of shape [num_nodes, num_nodes] containing path lengths
        :return: torch.Tensor, Edge Encoding matrix
        """
        device = next(self.parameters()).device
        cij = torch.zeros((x.shape[0], x.shape[0]), device=device)
        
        # Create mask to identify valid edges in the path tensor
        mask = edge_paths_tensor != -1
        # Use non-inplace operation, create a new tensor copy to avoid modifying the original tensor
        safe_edge_paths_tensor = edge_paths_tensor.clone()
        safe_edge_paths_tensor[~mask] = 0
        
        # Get edge attributes for all paths
        # Shape: [num_nodes, num_nodes, max_path_distance, edge_dim]
        edge_attrs = F.embedding(safe_edge_paths_tensor, edge_attr)
        
        # Expand edge_vector to match dimensions
        # Shape: [1, 1, max_path_distance, edge_dim]
        expanded_edge_vector = self.edge_vector.unsqueeze(0).unsqueeze(0)
        
        # Calculate dot product for each edge in each path
        # Shape: [num_nodes, num_nodes, max_path_distance]
        dot_products = (edge_attrs * expanded_edge_vector).sum(dim=-1)
        
        # Apply mask to ignore padding
        dot_products = dot_products * mask
        
        # Calculate average for each node pair, considering only valid paths
        # Sum over path length dimension, then divide by number of valid edges in each path
        valid_pairs = edge_paths_length > 0
        cij[valid_pairs] = dot_products.sum(dim=-1)[valid_pairs] / (edge_paths_length[valid_pairs] + 1e-10)
        
        cij = torch.nan_to_num(cij)
        
        return cij


class GraphormerAttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.edge_encoding = EdgeEncoding(edge_dim, max_path_distance)

        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch.Tensor,
                edge_paths_tensor,
                edge_paths_length,
                ptr=None) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial encoding
        :param edge_paths: pairwise node paths in edge indexes (deprecated)
        :param edge_paths_tensor: 3D tensor of shape [num_nodes, num_nodes, max_path_distance] containing edge indices
        :param edge_paths_length: 2D tensor of shape [num_nodes, num_nodes] containing path lengths
        :param ptr: pointer tensor for batching
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths: pairwise node paths in edge indexes (deprecated)
        :param edge_paths_tensor: 3D tensor of shape [num_nodes, num_nodes, max_path_distance] containing edge indices
        :param edge_paths_length: 2D tensor of shape [num_nodes, num_nodes] containing path lengths
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after attention operation
        """
        batch_mask_neg_inf = torch.full(size=(x.shape[0], x.shape[0]), fill_value=-1e6).to(
            next(self.parameters()).device)
        batch_mask_zeros = torch.zeros(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)

        # OPTIMIZE: get rid of slices: rewrite to torch
        if type(ptr) == type(None):
            batch_mask_neg_inf = torch.ones(size=(x.shape[0], x.shape[0])).to(next(self.parameters()).device)
            batch_mask_zeros += 1
        else:
            for i in range(len(ptr) - 1):
                batch_mask_neg_inf[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
                batch_mask_zeros[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1

        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        c = self.edge_encoding(x, edge_attr, edge_paths_tensor, edge_paths_length)
        a = self.compute_a(key, query, ptr)
        a = (a + b + c) * batch_mask_neg_inf
        softmax = torch.softmax(a, dim=-1) * batch_mask_zeros
        x = softmax.mm(value)
        return x

    def compute_a(self, key, query, ptr=None):
        if type(ptr) == type(None):
            a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5
        else:
            a = torch.zeros((query.shape[0], query.shape[0]), device=key.device)
            for i in range(len(ptr) - 1):
                a[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = query[ptr[i]:ptr[i + 1]].mm(
                    key[ptr[i]:ptr[i + 1]].transpose(0, 1)) / query.size(-1) ** 0.5

        return a


# FIX: PyG attention instead of regular attention, due to specificity of GNNs
class GraphormerMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int, edge_dim: int, max_path_distance: int):
        """
        :param num_heads: number of attention heads
        :param dim_in: node feature matrix input number of dimension
        :param dim_q: query node feature matrix input number dimension
        :param dim_k: key node feature matrix input number of dimension
        :param edge_dim: edge feature matrix number of dimension
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphormerAttentionHead(dim_in, dim_q, dim_k, edge_dim, max_path_distance) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch.Tensor,
                edge_paths_tensor=None,
                edge_paths_length=None,
                ptr=None) -> torch.Tensor:
        """
        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths_tensor: 3D tensor of shape [num_nodes, num_nodes, max_path_distance] containing edge indices
        :param edge_paths_length: 2D tensor of shape [num_nodes, num_nodes] containing path lengths
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after all attention heads
        """
        return self.linear(
            torch.cat([
                attention_head(x, edge_attr, b, edge_paths_tensor, edge_paths_length, ptr) for attention_head in self.heads
            ], dim=-1)
        )


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, n_heads, ff_dim, max_path_distance):
        """
        :param node_dim: node feature matrix input number of dimension
        :param edge_dim: edge feature matrix input number of dimension
        :param n_heads: number of attention heads
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_heads = n_heads
        self.ff_dim = ff_dim

        self.attention = GraphormerMultiHeadAttention(
            dim_in=node_dim,
            dim_k=node_dim,
            dim_q=node_dim,
            num_heads=n_heads,
            edge_dim=edge_dim,
            max_path_distance=max_path_distance,
        )
        self.ln_1 = nn.LayerNorm(self.node_dim)
        self.ln_2 = nn.LayerNorm(self.node_dim)
        self.ff = nn.Sequential(
                    nn.Linear(self.node_dim, self.ff_dim),
                    nn.GELU(),
                    nn.Linear(self.ff_dim, self.node_dim)
        )


    def forward(self,
                x: torch.Tensor,
                edge_attr: torch.Tensor,
                b: torch,
                edge_paths_tensor=None,
                edge_paths_length=None,
                ptr=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h′(l) = MHA(LN(h(l−1))) + h(l−1)
        h(l) = FFN(LN(h′(l))) + h′(l)

        :param x: node feature matrix
        :param edge_attr: edge feature matrix
        :param b: spatial Encoding matrix
        :param edge_paths_tensor: 3D tensor of shape [num_nodes, num_nodes, max_path_distance] containing edge indices
        :param edge_paths_length: 2D tensor of shape [num_nodes, num_nodes] containing path lengths
        :param ptr: batch pointer that shows graph indexes in batch of graphs
        :return: torch.Tensor, node embeddings after Graphormer layer operations
        """
        x_prime = self.attention(self.ln_1(x), edge_attr, b, edge_paths_tensor, edge_paths_length, ptr) + x
        x_new = self.ff(self.ln_2(x_prime)) + x_prime

        return x_new
