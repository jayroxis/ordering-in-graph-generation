import pickle
import json
import torch
from torch.utils.data import Dataset
from .misc import *


class CircuitSignalToRawFeaturesDataset(Dataset):
    def __init__(
            self, 
            data_file,
            input_dim=2,
            output_dim=9,
            max_num_nodes=6,
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
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data_dict = self.data[i]
        return data_dict['signal'], data_dict['node_attributes']