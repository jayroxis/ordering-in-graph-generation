import pickle
import json
import torch
from torch.utils.data import Dataset


class CircuitSignalToRawFeaturesDataset(Dataset):
    def __init__(
            self, 
            data_file,
            input_dim=2,
            output_dim=9,
            max_num_nodes=6,
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
        self.input_dim=128
        self.output_dim=9
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data_dict = self.data[i]
        return data_dict['signal'], data_dict['node_attributes']