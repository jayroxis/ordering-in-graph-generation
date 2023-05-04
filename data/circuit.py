import pickle
import json
import torch
from torch.utils.data import Dataset
from .misc import *


__all__ = [
    "CircuitSignalToRawFeaturesDataset",
]


# Mean and Std values for normalizing the data
SIGNAL_MIN = [-0.8909, -0.7992]
SIGNAL_MAX = [0.7910, 0.8415]
NODE_MIN = [0.0000, -197.9330,   50.0000,    6.0000,  -46.9990, -241.2470,
           2.4000,    2.4000,    0.0000]
NODE_MAX =[346.5820, 207.6280, 100.0000,   6.0000, 388.8330, 234.1580,   7.2500,
          7.2500,   1.0000]


class CircuitSignalToRawFeaturesDataset(Dataset):
    def __init__(
            self, 
            data_file,
            input_dim: int=2,
            output_dim: int=9,
            max_num_nodes: int=6,
            signal_min=SIGNAL_MIN,
            signal_max=SIGNAL_MAX,
            node_min=NODE_MIN,
            node_max=NODE_MAX,
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
        self.signal_min = torch.tensor(signal_min, dtype=torch.float32).unsqueeze(0)
        self.signal_max = torch.tensor(signal_max, dtype=torch.float32).unsqueeze(0)
        self.node_min = torch.tensor(node_min, dtype=torch.float32).unsqueeze(0)
        self.node_max = torch.tensor(node_max, dtype=torch.float32).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data_dict = self.data[i]
        signal = data_dict['signal']
        signal = (signal - self.signal_min) / (self.signal_max - self.signal_min + 1e-8)
        node_attributes = data_dict['node_attributes']
        node_attributes = (node_attributes - self.node_min) / (self.node_max - self.node_min + 1e-8)
        node_attributes = self.sort_func(node_attributes)
        return signal, node_attributes