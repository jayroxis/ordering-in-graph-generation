
# Planar graph datasets
from .planar_graph import RenderedPlanarGraphDataset
from .scene_graph import PSGRelationDataset
from .molecule import MolecularDatasetsOccur2SMILES, MolecularDatasetsAtoms2SMILES


__all__ = [
    "RenderedPlanarGraphDataset", "PSGRelationDataset",
    "MolecularDatasetsOccur2SMILES", "MolecularDatasetsAtoms2SMILES"
]


def build_dataset(dataset_name: str, params: dict = {}):
    """
    Unified entry function for building datasets. 
    """
    dataset = eval(dataset_name)(**params)
    return dataset