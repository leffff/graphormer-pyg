from typing import Dict

import torch


def node_path_matrix_form_dict(node_paths: Dict, num_nodes: int) -> torch.Tensor:
    distance_matrix = torch.zeros((num_nodes, num_nodes))

    for src in node_paths:
        for dst in node_paths[src]:
            path_len = len(node_paths[src][dst])
            distance_matrix[src][dst] = path_len

    return distance_matrix


def get_batch_mask(ptr: torch.LongTensor, num_nodes: int):
    batch_mask = torch.zeros((num_nodes, num_nodes))
    # OPTIMIZE: get rid of slices: rewrite to torch
    for i in range(len(ptr) - 1):
        batch_mask[ptr[i]:ptr[i + 1], ptr[i]:ptr[i + 1]] = 1
