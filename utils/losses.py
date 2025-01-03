import numpy as np
import torch.nn.functional as F
import torch as th
from torch_scatter import scatter
import torch_scatter
import torch

def ph_loss(input,target,c):
    assert input.shape == target.shape, "Input and target must have the same shape"
    return th.sqrt((input - target) ** 2 + c**2) - c


def mean_flat_graph(tensor, batch):
    """
    Take the mean over all non-batch dimensions for graph data.

    Args:
        tensor (torch.Tensor): Node or edge features of shape [num_nodes, feature_dim].
        batch (torch.Tensor): Batch vector of shape [num_nodes], associating elements to graphs.

    Returns:
        torch.Tensor: Mean value per graph, shape [num_graphs].
    """
    # Aggregate sum per graph using the batch index
    sum_per_graph = torch_scatter.scatter(tensor.sum(dim=-1), batch, dim=0, reduce="sum")

    # Count elements per graph
    count_per_graph = torch_scatter.scatter(torch.ones_like(tensor[:, 0]), batch, dim=0, reduce="sum")

    # Compute mean per graph
    mean_per_graph = sum_per_graph / count_per_graph
    return mean_per_graph
