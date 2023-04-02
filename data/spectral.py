import torch

def get_edge_adjacency(node_pairs):
    L = node_pairs.shape[0]

    # Reshape the node pairs tensor for broadcasting
    edges = node_pairs.view(L, 2, -1)

    # Compute node pair equality
    node_match = (edges.unsqueeze(1) == edges.unsqueeze(0))
    A = node_match.all(dim=-1).any(dim=-1)

    return A.long()


def get_midpoint_distance(node_pairs):
    L = node_pairs.shape[0]

    # Reshape the node pairs tensor for broadcasting
    edges = node_pairs.view(L, 2, -1)
    mid_point = edges.mean(1)

    # Compute node pair distance
    distance = (mid_point.unsqueeze(1) - mid_point.unsqueeze(0))
    distance = torch.linalg.norm(distance, dim=-1)
    return distance


def get_midpoint_similarity(node_pairs):
    distance = get_midpoint_distance(node_pairs)
    similarity = 1.0 / (1.0 + distance)
    return similarity


def get_combined_similarity(node_pairs, alpha=0.5):
    A = get_edge_adjacency(node_pairs)
    M = get_midpoint_similarity(node_pairs)
    S = alpha * A + (1 - alpha) * M
    S = S / (S.max() + 1e-9)
    return S


def spectral_sort(node_pairs):
    S = get_combined_similarity(node_pairs, alpha=0.5)
    D = torch.diag(S.sum(dim=1))
    L = D - S
    E, V = torch.linalg.eigh(L)

    # calculate inverse similarity (or distance) matrix
    inv_S = 1.0 / (S + 1e-9) - 1.0

    # project inverse similarity (or distance) matrix on laplacian eigenvectors
    P = inv_S @ V

    # use the mean from first and second eigenpairs 
    # (spectral gap and Fieler vector)
    W = P[:, :2].mean(1)
    idx = torch.argsort(W)
    return node_pairs[idx]