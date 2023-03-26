import cv2
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

import torch
import torch.utils.data as data
from torchvision import transforms

from .misc import *


def full_angle_to_arc_angle(full_angle_degrees):
    circumference = 2 * math.pi
    arc_length = (full_angle_degrees / 360) * circumference
    arc_angle_radians = arc_length
    return arc_angle_radians

def arc_angle_to_full_angle(arc_angle_radians):
    circumference = 2 * math.pi
    arc_length = arc_angle_radians
    full_angle_degrees = (arc_length / circumference) * 360
    return full_angle_degrees


class RandomPlanarGraphDataset(data.Dataset):
    def __init__(
        self, 
        num_samples, 
        num_points, 
        epsilon, 
        tiny_angle, 
        max_retries=10, 
        verbose=False,
        **kwargs,
    ):
        """
        Constructor for the RandomPlanarGraphDataset class.

        Args:
        - num_samples: the number of random planar graphs to generate
        - num_points: the number of points to use when generating each planar graph
        - epsilon: the distance parameter for the Poisson disk 
          sampling algorithm used to generate the points
        - tiny_angle: the minimum angle between any two edges in the planar graph
        - max_retries: the maximum number of times to attempt 
          to generate a planar graph before giving up
        - verbose: whether to print error message
        """
        self.num_samples = num_samples
        self.num_points = num_points
        self.epsilon = epsilon
        if tiny_angle > math.pi:
            tiny_angle = full_angle_to_arc_angle(tiny_angle)
            assert tiny_angle <= (math.pi / 2), f"`tiny angle`= {arc_angle_to_full_angle}' > 90'."
        self.tiny_angle = tiny_angle
        self.max_retries = max_retries
        self.verbose = verbose
        
    def __len__(self):
        """
        Return the number of random planar graphs in the dataset.
        """
        return self.num_samples

    def __getitem__(self, index):
        """
        Generate a random planar graph and return its edges and vertices.

        Args:
        - index: the index of the planar graph to generate

        Returns:
        - V: a torch.Tensor of shape (num_points, 2) containing 
          the node coordinates of the planar graph
        - E: a torch.Tensor of shape (2, num_edges) containing 
          the indices of the nodes that form each edge in the planar graph
        """
        retries = 0
        while retries < self.max_retries:
            try:
                # Generate planar graph
                pg = PlanarGraph(self.num_points, self.epsilon, self.tiny_angle)
                pg.generate_planar_graph()
                pg.collapse_edges()
                pg.remove_disconnected_components()
                pg.remove_self_loops()

                # Get edges and vertices
                V, E = extract_node_positions(pg.G)
                V = torch.tensor(V).float()
                E = torch.tensor(E).long()

                return V, E

            except Exception as e:
                # If there's an error generating the planar graph, try again
                retries += 1
                if self.verbose:
                    print(f"Failed to generate planar graph {index} "
                          f"(attempt {retries}/{self.max_retries}): {e}")

        # If we exceed the maximum number of retries, return None for both V and E
        print(f"Max retries exceeded for planar graph {index}")
        return None, None

    
    
    
class RenderedPlanarGraphDataset(RandomPlanarGraphDataset):
    """
    A class for generating rendered planar graphs as 
    images with node coordinate pairs as labels.

    Inherits from RandomPlanarGraphDataset.

    Args:
        - img_size: image resolution to render
        *args: Positional arguments to pass to the parent class constructor.
        **kwargs: Keyword arguments to pass to the parent class constructor.
    """
    def __init__(self, img_size=224, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = img_size

    def __getitem__(self, index):
        """
        Generate a rendered planar graph and corresponding 
        node coordinate pairs for a given index.

        Args:
            index (int): Index of the graph to generate.

        Returns:
            tuple: A tuple of the rendered image as a torch 
                   tensor and the node coordinate pairs as a torch tensor.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                # Generate planar graph
                pg = PlanarGraph(self.num_points, self.epsilon, self.tiny_angle)
                pg.generate_planar_graph()
                pg.collapse_edges()
                pg.remove_disconnected_components()
                pg.remove_self_loops()

                # Get edges and vertices
                V, E = extract_node_positions(pg.G)
                V = torch.tensor(V).float()
                E = torch.tensor(E).long()
                
                # Get node coordinate pairs
                node_pair = get_node_pairs_single(V, E)
                
                # Random Shuffle the node pairs
                node_pair = sort_by_columns(node_pair)
                
                # Get rendered image
                linewidth = np.random.randint(5, 10)
                nodesize = np.random.randint(3, 8)
                img = pg.render_graph(
                    edge_color=(0, 0, 0), 
                    node_color=(0, 0, 0), 
                    size=self.img_size, 
                    linewidth=linewidth,
                    nodesize=nodesize,
                )
                t = transforms.Compose([
                    transforms.ToTensor(),
                ])
                img = t(img)
                return img, node_pair

            except Exception as e:
                # If there's an error generating the planar graph, try again
                retries += 1
                if self.verbose:
                    print(f"Failed to generate planar graph {index} "
                          f"(attempt {retries}/{self.max_retries}): {e}")

        # If we exceed the maximum number of retries, return None for both V and E
        print(f"Max retries exceeded for planar graph {index}")
        return None, None



class LatentSortGraphDataset(RenderedPlanarGraphDataset):
    """
    Inherits from RenderedPlanarGraphDataset. This dataset class has latent sort
    incorporated.

    Latent sort is a technique to convert any unordered data structure, such as a
    graph, into an ordered sequence. This is done using an encoder that encodes the
    input sequence to a 1D latent sequence. By using `argsort` on the 1D latent
    sequence, a deterministic order can be obtained for any unordered data structure.
    This is crucial to convert any unordered data into an ordered sequence, which is
    required for training sequential models.

    Args:
        - encoder (torch.nn.Module): The latent sort encoder.
        *args: Positional arguments to pass to the parent class constructor.
        **kwargs: Keyword arguments to pass to the parent class constructor.
    
    Note:
        To get fast inference performance, when preparing latent sort encoder, it is
        recommended to use PyTorch JIT module.

        Here is an example code for saving the trained encoder model:

        ```python
        # Make sure the model is on cpu and set to eval mode.
        encoder = encoder.cpu().eval()

        # Enable OneDNN-Graph for fast inference
        torch.jit.enable_onednn_fusion(enabled=True)

        # Trace the model if it works, otherwise torch.jit.script
        dummy_input = [torch.rand(32, 4)] # dummy input
        traced_model = torch.jit.trace(encoder, dummy_input)
        traced_model = torch.jit.freeze(traced_model)

        # Disable OneDNN-Graph as default
        torch.jit.enable_onednn_fusion(enabled=False)

        ```

    """
    def __init__(self, encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)

        encoder = encoder.cpu().eval()
        self.encoder = encoder

    @torch.no_grad()
    def __getitem__(self, index):
        """
        Generate a rendered planar graph and corresponding node coordinate pairs for a
        given index.

        Args:
            index (int): Index of the graph to generate.

        Returns:
            tuple: A tuple of the rendered image as a torch tensor and the node
            coordinate pairs as a torch tensor.
        """
        img, node_pair = super().__getitem__(index)
        emb = self.encoder(node_pair)
        sort_idx = emb.argsort(0).squeeze(1)
        node_pair = node_pair[sort_idx]
        return img, node_pair




class PlanarGraph:
    """
    A class representing a planar graph generated from a set of random points.

    Parameters:
    - num_points (int): the number of random points to generate
    - epsilon (float): the minimum distance between any two points
    - tiny_angle (float): the minimum angle between two edges incident to a vertex
    """
    def __init__(self, num_points: int, epsilon: float, tiny_angle: float):
        """
        Constructor for PlanarGraph class.
        
        Args:
        - num_points (int): the number of random points to generate
        - epsilon (float): the minimum distance between any two points
        - tiny_angle (float): the minimum angle between two edges incident to a vertex
        """
        self.num_points = num_points
        self.epsilon = epsilon
        self.tiny_angle = tiny_angle
        self.points = None
        self.G = None

    def generate_points(self) -> None:
        """
        Generate random points and check distances
        """
        points = np.random.rand(self.num_points, 2)
        points = self.check_point_distances(points)
        self.points = points

    def check_point_distances(self, points: np.ndarray) -> np.ndarray:
        """
        Checks distances between points and replaces any points that are too close together.
        
        Args:
        - points (numpy array): array of (x,y) coordinates of the points
        
        Returns:
        - numpy array: array of (x,y) coordinates of the points with no points closer than epsilon
        """
        # Check distances between points
        distances = np.linalg.norm(points[:, np.newaxis, :] - points, axis=2)
        np.fill_diagonal(distances, np.inf)
        while np.any(distances < self.epsilon):
            mask = distances < self.epsilon
            rows, _ = np.where(mask)
            num_points_to_change = len(rows)
            new_points = np.random.rand(num_points_to_change, 2)
            points[rows] = new_points
            distances = np.linalg.norm(points[:, np.newaxis, :] - points, axis=2)
            np.fill_diagonal(distances, np.inf)
        return points

    def generate_planar_graph(self) -> None:
        """
        Generates a planar graph from a set of random points using Delaunay triangulation.
        """
        self.generate_points()

        # Compute Delaunay triangulation
        tri = Delaunay(self.points)

        # Iterate over edges and remove ones shorter than epsilon
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    p1, p2 = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add((p1, p2))

        # Create networkx graph
        G = nx.Graph()
        for p1, p2 in edges:
            length = np.linalg.norm(self.points[p1] - self.points[p2])
            G.add_edge(p1, p2, weight=length)

        # Add coordinates as node attributes
        for i in range(self.num_points):
            G.nodes[i]['pos'] = self.points[i]

        self.G = G
    
    def collapse_edges(self) -> None:
        """
        Collapses edges in the graph that have a small angle between them.
        """
        edges_to_remove = set()

        for edge in self.G.edges():
            p1, p2 = edge
            neighbors_p1 = set(self.G.neighbors(p1)) - {p2}
            neighbors_p2 = set(self.G.neighbors(p2)) - {p1}

            for p3 in neighbors_p1:
                v1, v2 = self.points[p1] - self.points[p2], self.points[p1] - self.points[p3]
                len_v1, len_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
                eps = 1e-8
                cos_angle = np.dot(v1, v2) / max((len_v1 * len_v2), eps)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                if angle < self.tiny_angle:
                    edges_to_remove.add((p1, p2))
                    break

            for p3 in neighbors_p2:
                v1, v2 = self.points[p2] - self.points[p1], self.points[p2] - self.points[p3]
                len_v1, len_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
                eps = 1e-8
                cos_angle = np.dot(v1, v2) / max((len_v1 * len_v2), eps)
                angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                if angle < self.tiny_angle:
                    edges_to_remove.add((p1, p2))
                    break

        for edge in edges_to_remove:
            if self.G.has_edge(*edge):
                self.G.remove_edge(*edge)

    def remove_disconnected_components(self) -> None:
        """
        Removes any disconnected components in the graph.
        """
        # Relabel nodes with unique labels
        node_mapping = {n: i for i, n in enumerate(self.G.nodes)}
        self.G = nx.relabel_nodes(self.G, node_mapping)

        # Remove disconnected components
        subgraphs = list(nx.connected_components(self.G))
        largest_subgraph = max(subgraphs, key=len)
        nodes_to_remove = set(self.G.nodes) - largest_subgraph

        # Remove edges connected to removed nodes
        edges_to_remove = []
        for node1, node2 in self.G.edges:
            if node1 in nodes_to_remove or node2 in nodes_to_remove:
                edges_to_remove.append((node1, node2))

        self.G.remove_edges_from(edges_to_remove)
        self.G.remove_nodes_from(nodes_to_remove)

        # Reset node indices and update edges
        node_mapping = {old: new for new, old in enumerate(largest_subgraph)}
        nx.relabel_nodes(self.G, node_mapping, copy=False)

        # Update points to match remaining nodes
        self.points = np.array([self.G.nodes[n]['pos'] for n in self.G])
        
    def remove_self_loops(self) -> None:
        """
        Removes any self-loops in the graph.
        """
        nodes_with_selfloops = [n for n in self.G.nodes if self.G.has_edge(n, n)]
        selfloops = [(n, n) for n in nodes_with_selfloops]
        self.G.remove_edges_from(selfloops)
        
    def plot_graph(self, save_path: str = None) -> None:
        """
        Plots the planar graph using matplotlib or saves to file if save_path is specified.

        Args:
        - save_path (string): the path to save the plot to as a file (default is None)
        """
        fig, ax = plt.subplots()
        pos = nx.get_node_attributes(self.G, 'pos')
        nx.draw(self.G, pos=pos, ax=ax, font_color='white')
        nx.draw_networkx_labels(
            self.G, 
            pos, 
            font_size=10, 
            font_family="sans-serif", 
            font_color='white'
        )
        plt.show()
    
    def render_graph(self, 
            edge_color: tuple = (0, 0, 0), 
            node_color: tuple = (0, 255, 0), 
            size: int = 400, 
            linewidth: int = 2,
            nodesize: int = 2,
        ) -> np.ndarray:
        """
        Renders the planar graph as an image using OpenCV.

        Args:
        - edge_color (tuple): the color of the edges in the rendered image as a tuple of three integers 
                              (default is black)
        - node_color (tuple): the color of the nodes in the rendered image as a tuple of three integers 
                              (default is green)
        - size (int): the size of the rendered image (default is 400)
        - linewidth (int): the width of the edges in the rendered image (default is 2)

        Returns:
        - numpy array: the rendered image as a 3-channel RGB array
        """
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        pos = nx.get_node_attributes(self.G, 'pos')

        for node1, node2 in self.G.edges:
            p1, p2 = pos[node1], pos[node2]
            pt1, pt2 = tuple((p1 * size).astype(int)), tuple((p2 * size).astype(int))
            cv2.line(img, pt1, pt2, edge_color, linewidth)

        for i in pos.keys():
            pt = tuple((pos[i] * size).astype(int))
            cv2.circle(img, pt, nodesize, node_color, -1)

        return img
    
    def test_run(self) -> None:
        """
        Runs a test sequence on the PlanarGraph object, generating a 
        planar graph and applying various transformations to it.
        """
        self.generate_planar_graph()
        self.collapse_edges()
        self.remove_disconnected_components()
        self.remove_self_loops()
        self.plot_graph()
