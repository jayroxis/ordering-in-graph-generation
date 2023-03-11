
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import networkx as nx



class PadSequence:
    def __init__(self, pad_value=-1):
        self.pad_value = pad_value
    
    def _pad_sequence(self, seq):
        seq_padded = pad_sequence(
            seq, 
            batch_first=True, 
            padding_value=self.pad_value
        )
        return seq_padded
        
    def __call__(self, batch):
        transposed = list(map(list, zip(*batch)))
        padded = []
        for items in transposed:
            padded.append(self._pad_sequence(items))
        return tuple(padded)

    
    
def extract_node_positions(G):
    """
    Extracts the node positions and edges of a NetworkX graph.

    Parameters:
    - G: a NetworkX graph object.

    Returns:
    - V: a 2D numpy array of node positions, where each row corresponds to a node in the graph.
    - E: a 2D numpy array of edge indices, where each row corresponds to an edge in the graph.
    """
    # Get the 'pos' attribute of all nodes in the graph
    pos_dict = nx.get_node_attributes(G, 'pos')
    # Sort the node IDs in ascending order
    node_list = sorted(pos_dict.keys())
    # Get the number of nodes in the graph
    num_nodes = len(node_list)
    # Create an empty list to hold the node positions
    V = []
    # Iterate over the sorted node IDs
    for i, node in enumerate(node_list):
        # Append the node position to the list
        V.append(pos_dict[node])
    # Stack the node positions into a 2D numpy array
    V = np.stack(V)
    # Get the edges of the graph as pairs of node IDs
    E = np.array(G.edges())
    # Convert the node IDs to their corresponding indices in V
    E = np.array([(node_list.index(u), node_list.index(v)) for u, v in E]).T
    # Return the node positions and edge list
    return V, E    

    

# def sort_by_columns(E):
#     """
#     Sorts the input tensor E along its columns, based on the minimum
#     value of each row across two equal-sized submatrices of E.
    
#     Args:
#         E (torch.Tensor): Input tensor to be sorted. Must have shape (n_samples, n_features).
    
#     Returns:
#         sorted_E (torch.Tensor): Sorted tensor, with the same shape as E.
#     """
#     # Make a copy of the input tensor E and store its original data type
#     orig_E = E.clone()
#     orig_dtype = E.dtype
    
#     # Divide E into two equal-sized submatrices
#     half_target_dim = int(E.shape[-1] // 2)
#     E1 = E[..., :half_target_dim]
#     E2 = E[..., half_target_dim:]
    
#     # Compute the element-wise minimum of E1 and E2
#     E = E1 + E2
    
#     # Convert the tensor to a numpy array and use np.lexsort to get the sorted indices
#     E = E.numpy()
#     indices = np.lexsort([E[:, i] for i in range(E.shape[1]-1, -1, -1)])
    
#     # Convert the sorted indices back to a torch tensor and return it
#     sorted_E = torch.as_tensor(orig_E[indices], dtype=orig_dtype)
#     return sorted_E


def sort_by_columns(E, ascending=False):
    """
    Sorts the input tensor E based on the root mean square value of each row.
    
    Args:
        E (torch.Tensor): Input tensor to be sorted. Must have shape (n_samples, n_features).
        ascending (bool): Whether to sort in ascending or descending order. Default is False (descending).
    
    Returns:
        sorted_E (torch.Tensor): Sorted tensor, with the same shape as E.
    """
    # Make a copy of the input tensor E and store its original data type
    orig_E = E.clone()
    orig_dtype = E.dtype
    
    # Compute the root mean square value of each row of E
    rms = torch.sqrt(torch.mean(E ** 2, dim=-1))
 
    # Convert the tensor to a numpy array and use np.argsort to get the sorted indices
    rms = rms.numpy()
    indices = np.argsort(rms) if ascending else np.argsort(-rms)
    
    # Convert the sorted indices back to a torch tensor and return it
    sorted_E = torch.as_tensor(orig_E[indices], dtype=orig_dtype)
    return sorted_E



def get_node_pairs_batch(V, E):
    """
    Given a set of node coordinates V and a set of edges E,
    return the node coordinate pairs that correspond to the edges in E.
    
    Args:
    - V: a torch.Tensor of shape (batch_size, num_nodes, num_dimensions) 
      containing node coordinates
    - E: a torch.Tensor of shape (batch_size, 2, num_edges) 
      containing edge index pairs
    
    Returns:
    - edge_pairs: a torch.Tensor of shape (batch_size, num_edges, num_dimensions, 2) 
      containing node coordinate pairs
    """
    # get the start and end nodes of each edge
    start_nodes = E[:, 0]
    end_nodes = E[:, 1]
    
    # use the start and end nodes to extract the corresponding node coordinates
    start_coords = torch.gather(V, 1, start_nodes.unsqueeze(-1).repeat(1, 1, V.shape[-1]))
    end_coords = torch.gather(V, 1, end_nodes.unsqueeze(-1).repeat(1, 1, V.shape[-1]))
    
    # concatenate the start and end coordinates to get the final output tensor
    edge_pairs = torch.cat([start_coords, end_coords], dim=-1)
    
    return edge_pairs



def get_node_pairs_single(V, E):
    """
    Given a set of node coordinates V and a set of edges E,
    return the node coordinate pairs that correspond to the edges in E.
    
    Args:
    - V: a torch.Tensor of shape (num_nodes, num_dimensions) 
      containing node coordinates
    - E: a torch.Tensor of shape (2, num_edges) containing edge index pairs
    
    Returns:
    - edge_pairs: a torch.Tensor of shape (num_edges, num_dimensions, 2) 
      containing node coordinate pairs
    """
    # get the start and end nodes of each edge
    start_nodes = E[0]
    end_nodes = E[1]
    
    # use the start and end nodes to extract the corresponding node coordinates
    start_coords = V[start_nodes]
    end_coords = V[end_nodes]
    
    # concatenate the start and end coordinates to get the final output tensor
    edge_pairs = torch.cat([start_coords, end_coords], dim=-1)
    
    return edge_pairs


def get_node_pairs(V, E):
    """
    Given a set of node coordinates V and a set of edges E,
    return the node coordinate pairs that correspond to the edges in E.
    
    Args:
    - V: a torch.Tensor of shape (num_nodes, num_dimensions) or 
      (batch_size, num_nodes, num_dimensions) containing node coordinates
    - E: a torch.Tensor of shape (2, num_edges) or 
      (batch_size, 2, num_edges) containing edge index pairs
    
    Returns:
    - edge_pairs: a torch.Tensor of shape (num_edges, num_dimensions, 2) 
      or (batch_size, num_edges, num_dimensions, 2) containing node coordinate pairs
    """
    if len(V.shape) == 2 and len(E.shape) == 2:
        # single graph
        return get_node_pairs_single(V, E)
    elif len(V.shape) == 3 and len(E.shape) == 3:
        # minibatch of graphs
        return get_node_pairs_batch(V, E)
    else:
        raise ValueError("Invalid input shapes: V={}, E={}".format(V.shape, E.shape))


        
