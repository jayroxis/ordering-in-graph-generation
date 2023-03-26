
# Planar graph datasets
from .planar_graph import *


def build_dataset(dataset_name: str, *args, **params):
    """
    Unified entry function for building datasets. 
    """
    dataset = eval(dataset_name)(*args, **params)
    return dataset