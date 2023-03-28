
import os 
from torch.utils.data import Dataset


class MolecularDatasetsOccur2SMILES(Dataset):
    def __init__(self, atom_file: str, smiles_file: str):
        super().__init__()
        atoms = np.load(atom_file, allow_pickle=True)
        smiles = np.load(smiles_file, allow_pickle=True)
        atom_occur = [a.sum(0) for a in atoms]
        atom_occur = np.stack(atom_occur)