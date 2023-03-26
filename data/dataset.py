
# Planar graph datasets
from .planar_graph import *


def build_dataset(dataset_name: str, params: dict = {}):
    """
    Unified entry function for building datasets. 
    """
    dataset = eval(dataset_name)(**params)
    return dataset