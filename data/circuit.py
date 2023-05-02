import pickle
import json
import torch
from torch.utils.data import Dataset
from .misc import *


__all__ = [
    "CircuitSignalToRawFeaturesDataset",
]


SIGNAL_MEAN = [-0.0405, -0.0013]
SIGNAL_STD = [0.1421, 0.1428]
NODE_MEAN =[6.3346e+01, 9.6438e-04, 7.3591e+01, 6.0000e+00, 6.3395e+01, 4.9361e-02,
        4.5948e+00, 4.5912e+00, 4.9970e-01]
NODE_STD = [59.2894, 55.0359, 14.3586,  1.0e-8, 65.1669, 61.3278,  2.2104,  2.2103,
         0.2889]


class CircuitSignalToRawFeaturesDataset(Dataset):
    def __init__(
            self, 
            data_file,
            input_dim: int=2,
            output_dim: int=9,
            max_num_nodes: int=6,
            signal_mean=SIGNAL_MEAN,
            signal_std=SIGNAL_STD,
            node_mean=NODE_MEAN,
            node_std=NODE_STD,
            sort_func="no_sort",
            *args, **kwargs
        ):
        """
        This class returns the raw parameters
        The GNN attributes are calculated in a rule-based way from the 9 raw attributes
        Might make more sense for us to predict the raw attributes
        Except possibly x and y. Because they mention in the paper that it is the relative distance that matters.
        """
        super().__init__(*args, **kwargs)
        if data_file.endswith('.pkl'):
            self.data = pickle.load(open(data_file, 'rb'))
        elif data_file.endswith('.json'):
            self.data = json.load(open(data_file, 'r'))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_num_nodes = max_num_nodes

        if type(sort_func) == str:
            sort_func = eval(sort_func)
        elif type(sort_func) == dict:
            sort_func = eval(sort_func["class"])(**sort_func["params"])
        self.sort_func = sort_func
        self.signal_mean = torch.tensor(signal_mean, dtype=torch.float32).unsqueeze(0)
        self.signal_std = torch.tensor(signal_std, dtype=torch.float32).unsqueeze(0)
        self.node_mean = torch.tensor(node_mean, dtype=torch.float32).unsqueeze(0)
        self.node_std = torch.tensor(node_std, dtype=torch.float32).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data_dict = self.data[i]
        signal = data_dict['signal']
        signal = (signal - self.signal_mean) / self.signal_std
        node_attributes = data_dict['node_attributes']
        node_attributes = self.sort_func(node_attributes)
        node_attributes = (node_attributes - self.node_mean) / self.node_std
        return signal, node_attributes