import networkx as nx
import torch
#from torch_geometric.data import Data
import random
import numpy as np

def prioritize_by_eigenvector_centrality(edge_index, num_nodes):
    """
    Prioritize nodes based on eigenvector centrality.

    Args:
        edge_index (torch.Tensor): Edge index of the graph (shape [2, num_edges]).
        num_nodes (int): Number of nodes in the graph.

    Returns:
        torch.Tensor: Tensor of node indices sorted by eigenvector centrality.
    """
    # Create a NetworkX graph from edge_index
    G = nx.Graph()
    edges = edge_index.T.cpu().numpy()
    G.add_edges_from(edges)

    # Compute eigenvector centrality
    centrality = nx.eigenvector_centrality_numpy(G)

    # Get centrality scores and sort nodes
    centrality_scores = torch.tensor(
        [centrality[i] for i in range(num_nodes)], dtype=torch.float
    )
    prioritized_nodes = torch.argsort(centrality_scores, descending=True)

    return prioritized_nodes

