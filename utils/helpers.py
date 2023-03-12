
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np



def shuffle_tensor(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Shuffle the tensor along the specified dimension independently 
    for each slice along that dimension.
    
    [NOTE] this is different to torch.perm which is shared permutation ordering.
    
    For example, for a 2D tensor:
        x = tensor([[0, 0, 1, 6, 2],
                    [5, 4, 7, 2, 4]])
            
    Aftering shuffling `x` becomes:
            tensor([[0, 0, 6, 2, 1],
                    [5, 7, 2, 4, 4]])
                    
    This works for any given dimension `dim` for an arbitary shape input tensor `x`.
    
    Args:
        x (torch.Tensor): The tensor to shuffle.
        dim (int, optional): The dimension along which to shuffle. Default is 0.

    Returns:
        torch.Tensor: The shuffled tensor.
    """
    # Get the shape of the slice of x along the specified dimension
    x_shape = tuple(x.shape[:dim+1])

    # Get the indices that would sort a random tensor along the specified dimension
    indices = torch.argsort(torch.rand(x_shape, device=x.device), dim=dim)
    
    # Add singleton dimensions to indices to match the shape of x
    n = x.dim() - dim - 1   
    indices = indices[(..., ) + (None,) * n] 
    indices = indices * torch.ones_like(x)
    indices = indices.long()
    
    # Shuffle the tensor using gather
    shuffled_x = torch.gather(x, dim=dim, index=indices)
    return shuffled_x



def unpad_node_pairs(node_pair, padding_value=-1.0, error=0.1):
    """
    Truncates the padding values at the end of the node_pair tensor.
    Args:
        node_pair: a PyTorch tensor or a NumPy array of shape (N, 4),
            where the N is the number of edges in the graph to plot.
            The 4 dimensional vector for each node pair, the first 2
            dimensions and the last 2 dimensions are the (x, y)
            coordinate of the two vertices of an edge in the graph.
            The coordinates range is (0, 1) at each dimension for the plot.
            The last row(s) of the tensor may contain padding values,
            which should be truncated from the tensor before plotting the graph.
        padding_value: a float value between -1 and 1, representing the minimum value
            for a dimension to be considered as padding. Default is -1.0.
        error: a float value representing the maximum absolute difference between
            a dimension value and the padding_value value for the dimension to be
            considered as padding. Default is 0.1.
    Returns:
        A new tensor with the padding values truncated from the end.
    """
    # Find the index of the last row that does not contain padding values
    last_nonpadding_row = 0
    for i in range(node_pair.shape[0]):
        if np.all(np.abs(node_pair[i, 2:] - padding_value) > error):
            last_nonpadding_row = i

    # Truncate the tensor to remove the padding values
    node_pair = node_pair[:last_nonpadding_row+1, :]

    return node_pair



def create_graph(node_pair, threshold=0.01):
    """
    Creates a networkx graph object from the input node_pair.

    Args:
        node_pair: a PyTorch tensor or a NumPy array of shape (N, 4),
            where the N is the number of edges in the graph to plot.
            The 4 dimensional vector for each node pair, the first 2
            dimensions and the last 2 dimensions are the (x, y)
            coordinate of the two vertices of an edge in the graph.
            The coordinates range is (0, 1) at each dimension for the plot.
        threshold: a float value representing the maximum distance between two nodes
            for them to be considered the same node. Default is 0.01.

    Returns:
        A networkx graph object.
    """
    # Create a dictionary of node positions
    node_pos = {}
    node_count = 0
    for i in range(node_pair.shape[0]):
        x1, y1, x2, y2 = node_pair[i]
        pos1 = (x1, y1)
        pos2 = (x2, y2)

        # Check if the positions are close to any existing positions
        found_match = False
        for node, pos in node_pos.items():
            if np.linalg.norm(np.array(pos) - np.array(pos1)) <= threshold:
                found_match = True
                break
        if not found_match:
            node_pos[node_count] = pos1
            node_count += 1

        found_match = False
        for node, pos in node_pos.items():
            if np.linalg.norm(np.array(pos) - np.array(pos2)) <= threshold:
                found_match = True
                break
        if not found_match:
            node_pos[node_count] = pos2
            node_count += 1
            
    # Create a graph object
    G = nx.Graph()

    # Add edges to the graph
    for i in range(node_pair.shape[0]):
        x1, y1, x2, y2 = node_pair[i]
        pos1 = (x1, y1)
        pos2 = (x2, y2)

        node1 = None
        node2 = None
        for node, pos in node_pos.items():
            if np.linalg.norm(np.array(pos) - np.array(pos1)) <= threshold:
                node1 = node
            if np.linalg.norm(np.array(pos) - np.array(pos2)) <= threshold:
                node2 = node
        if node1 is not None and node2 is not None:
            G.add_edge(node1, node2)

    # Set node positions
    nx.set_node_attributes(G, node_pos, 'pos')

    return G



def draw_graph(G, save_path=None):
    """
    Draws a networkx graph using matplotlib or saves to file if save_path is specified.

    Args:
        G: a networkx graph object.
        save_path (string): the path to save the plot to as a file (default is None)
    """
    # Draw the graph
    pos = nx.get_node_attributes(G, 'pos')
    fig, ax = plt.subplots()
    nx.draw(G, pos=pos, ax=ax, font_color='white')
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", font_color='white')

    # Save or show the plot
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

        
def plot_graph(node_pair, save_path=None, threshold=0.01):
    """
    Plots a graph with node coordinates from the input node_pair using matplotlib or saves to file
    if save_path is specified.

    Args:
        node_pair: a PyTorch tensor or a NumPy array of shape (N, 4),
            where the N is the number of edges in the graph to plot.
            The 4 dimensional vector for each node pair, the first 2
            dimensions and the last 2 dimensions are the (x, y)
            coordinate of the two vertices of an edge in the graph.
            The coordinates range is (0, 1) at each dimension for the plot.
        save_path (string): the path to save the plot to as a file (default is None)
    """
    G = create_graph(node_pair, threshold=threshold)
    draw_graph(G, save_path=save_path)


def plot_graph_strict(node_pair, save_path=None):
    """
    Plots a graph with node coordinates from the input node_pair using matplotlib or saves to file
    if save_path is specified.

    Args:
        node_pair: a PyTorch tensor or a NumPy array of shape (N, 4),
            where the N is the number of edges in the graph to plot.
            The 4 dimensional vector for each node pair, the first 2
            dimensions and the last 2 dimensions are the (x, y)
            coordinate of the two vertices of an edge in the graph.
            The coordinates range is (0, 1) at each dimension for the plot.
        save_path (string): the path to save the plot to as a file (default is None)
    """
    # Create a graph object
    G = nx.Graph()

    # Add edges to the graph
    for i in range(node_pair.shape[0]):
        # Extract the coordinates of the two vertices for each edge
        x1, y1, x2, y2 = node_pair[i]

        # Add the two vertices as nodes to the graph
        G.add_node(i, pos=(x1, y1))
        G.add_node(i+1, pos=(x2, y2))

        # Add the edge to the graph
        G.add_edge(i, i+1)

    # Draw the graph
    pos = nx.get_node_attributes(G, 'pos')
    fig, ax = plt.subplots()
    nx.draw(G, pos=pos, ax=ax, font_color='white')
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif", font_color='white')

    # Save or show the plot
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def format_time(seconds):
    """Converts a duration in seconds to a string in the format hh:mm:ss."""
    h = int(seconds / 3600)
    m = int((seconds % 3600) / 60)
    s = int(seconds % 60)
    return "{:02d}:{:02d}:{:02d}".format(h, m, s)