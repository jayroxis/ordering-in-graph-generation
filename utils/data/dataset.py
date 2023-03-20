import torch
import numpy as np
import torch.utils.data as data
from torchvision import transforms

from .planar_graph import PlanarGraph
from .misc import *
from .spectral import spectral_sort


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




class SpectralSortGraphDataset(RenderedPlanarGraphDataset):
    """
    Inherits from RenderedPlanarGraphDataset. This dataset class uses spectral
    sort.

    Args:
        *args: Positional arguments to pass to the parent class constructor.
        **kwargs: Keyword arguments to pass to the parent class constructor.

        ```

    """

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
        node_pair = spectral_sort(node_pair)
        return img, node_pair
