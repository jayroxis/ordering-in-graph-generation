

import torch
import numpy as np 
from torch.utils.data import Dataset
from torch.nn import functional as F


# Dictionary to encode SMILES: '_' represents [STOP_TOKEN]
smiles_dict = {
    '_': 0,  '#': 1,  '(': 2,  ')': 3,  '-': 4,  '1': 5,   
    '2': 6,  '3': 7,  '4': 8,  '5': 9,  '6': 10, '=': 11,  
    'B': 12, 'C': 13, 'F': 14, 'H': 15, 'N': 16, 'O': 17,  
    'S': 18, '[': 19, ']': 20, 'c': 21, 'l': 22, 'n': 23,  
    'o': 24, 'r': 25, 's': 26   
}


class MolecularDatasetsOccur2SMILES(Dataset):
    """
    Molecule dataset: Atom Occurrences to SMILES graph prediction.
    """
    def __init__(
        self, 
        atom_file:  str, 
        smiles_file: str, 
        occur_cls: int = 25,
        smile_cls: int = 27,
        one_hot: bool = True,
        **kwargs,
    ):
        super().__init__()
        atoms = np.load(atom_file, allow_pickle=True)
        self.smiles = np.load(smiles_file, allow_pickle=True)

        # Build atom occurrences
        atom_occur = [a.sum(0) for a in atoms]
        atom_occur = np.stack(atom_occur)
        self.atom_occur = torch.from_numpy(atom_occur)

        # register variables
        self.one_hot = one_hot
        self.occur_cls = occur_cls
        self.smile_cls = smile_cls

    def __len__(self):
        return len(self.atom_occur)

    def __getitem__(self, idx):
        occur = self.atom_occur[idx]
        label = [smiles_dict[s] for s in self.smiles[idx]]
        label = torch.tensor(label, dtype=torch.long)
        if self.one_hot:
            occur = F.one_hot(occur, num_classes=self.occur_cls).float() 
            label = F.one_hot(label, num_classes=self.smile_cls).float()
        return occur, label