
# Planar graph datasets
from .planar_graph import RenderedPlanarGraphDataset
from .scene_graph import PSGRelationDataset
from .Molecule import MolecularDatasetsOccur2SMILES


__all__ = [
    "RenderedPlanarGraphDataset", "PSGRelationDataset"
]


def build_dataset(dataset_name: str, params: dict = {}):
    """
    Unified entry function for building datasets. 
    """
    dataset = eval(dataset_name)(**params)
    return dataset