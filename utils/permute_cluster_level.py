import networkx as nx
import torch
import community.community_louvain as community_louvain

def permute_by_clusters(edge_index, num_nodes):
    """
    Permute nodes based on clusters defined by the Louvain algorithm.

    Args:
        edge_index (torch.Tensor): Edge index of the graph (shape [2, num_edges]).
        num_nodes (int): Number of nodes in the graph.

    Returns:
        torch.Tensor: Permuted indices of nodes.
    """
    # Convert to NetworkX graph
    G = nx.Graph()
    edges = edge_index.T.cpu().numpy()
    G.add_edges_from(edges)

    # Perform clustering using the Louvain algorithm
    partition = community_louvain.best_partition(G)

    # Group nodes by clusters
    clusters = {}
    for node, cluster_id in partition.items():
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(node)

    # Shuffle nodes within each cluster and concatenate
    permuted_indices = []
    for cluster_nodes in clusters.values():
        permuted_indices += torch.tensor(cluster_nodes)[torch.randperm(len(cluster_nodes))].tolist()

    return torch.tensor(permuted_indices)

# Example usage
n_nodes = 31
n_edges = 66
edge_index = torch.randint(0, n_nodes, (2, n_edges))
permuted_nodes = permute_by_clusters(edge_index, n_nodes)
print(permuted_nodes)
