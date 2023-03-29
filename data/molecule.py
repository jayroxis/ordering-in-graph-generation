
import os 
from torch.utils.data import Dataset
from torch.nn import functional as F


# Dictionary to encode SMILES
smiles_dict = {
    '#': 0,  '(': 1,  ')': 2,  '-': 3,  '1': 4,   '2': 5,  
    '3': 6,  '4': 7,  '5': 8,  '6': 9,  '=': 10,  'B': 11,  
    'C': 12, 'F': 13, 'H': 14, 'N': 15, 'O': 16,  'S': 17,  
    '[': 18, ']': 19, 'c': 20, 'l': 21, 'n': 22,  'o': 23,  
    'r': 24, 's': 25, '_': 26  # '_' represents [STOP_TOKEN]
}


class MolecularDatasetsOccur2SMILES(Dataset):
    """
    Molecule dataset: Atom Occurrences to SMILES graph prediction.
    """
    def __init__(
        self, 
        atom_file: str, 
        smiles_file: str, 
        occur_cls: int = 25,
        smile_cls: int = 27,
        one_hot: bool = True,
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