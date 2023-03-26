import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Delaunay


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
