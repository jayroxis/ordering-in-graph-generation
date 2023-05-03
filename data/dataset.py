
# Planar graph datasets
from .planar_graph import RenderedPlanarGraphDataset
from .scene_graph import PSGRelationDataset, PSGTRDataset
from .molecule import MolecularDatasetsOccur2SMILES, MolecularDatasetsAtoms2SMILES
from .circuit import CircuitSignalToRawFeaturesDataset
from .road_network import ToulouseRoadNetworkDataset

__all__ = [
    "RenderedPlanarGraphDataset", "PSGRelationDataset", "PSGTRDataset",
    "MolecularDatasetsOccur2SMILES", "MolecularDatasetsAtoms2SMILES",
    "CircuitSignalToRawFeaturesDataset", "ToulouseRoadNetworkDataset",
]


def build_dataset(dataset_name: str, params: dict = {}):
    """
    Unified entry function for building datasets. 
    """
    dataset = eval(dataset_name)(**params)
    return dataset