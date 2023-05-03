import pickle
import torch
from torch.utils.data import Dataset
from .misc import *
import numpy as np
from torchvision import transforms


__all__ = [
    "ToulouseRoadNetworkDataset",
]


class ToulouseRoadNetworkDataset(Dataset):
    def __init__(
            self, 
            img_file,
            label_file,
            img_size=64,
            train=False,
            sort_func="dfs_sort",
            remove_artificial_edges=True,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        
        self.label = pickle.load(open(label_file, 'rb'))
        self.label = {int(k): v for k, v  in self.label.items()}

        self.img_data = pickle.load(open(img_file, 'rb'))
        self.img_data = {int(k): v for k, v  in self.img_data.items()}
        
        self.train = train
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
        ])
        
        if sort_func == "bfs_sort":
            self.data = [(
                self.img_data[k], self.label[k]['bfs_edges']
            ) for k in self.img_data.keys()]
            self.sort_func = lambda x: x
        elif sort_func == "dfs_sort":
            self.data = [(
                self.img_data[k], self.label[k]['dfs_edges']
            ) for k in self.img_data.keys()]
            self.sort_func = lambda x: x
        elif type(sort_func) == str:
            self.data = [(
                self.img_data[k], self.label[k]['dfs_edges']
            ) for k in self.img_data.keys()]
            self.sort_func = eval(sort_func)
        elif type(sort_func) == dict:
            self.data = [(
                self.img_data[k], self.label[k]['dfs_edges']
            ) for k in self.img_data.keys()]
            self.sort_func = eval(sort_func["class"])(**sort_func["params"])
        else:
            self.data = [(
                self.img_data[k], self.label[k]['dfs_edges']
            ) for k in self.img_data.keys()]
            self.sort_func = sort_func

        # remove the edges added during BFS or DFS
        self.remove_artificial_edges = remove_artificial_edges

    def _remove_artificial_edges(self, seq):
        mask1 = (seq[..., :2] == 0.5).all(-1)
        mask2 = (seq[..., 2:] == 0.5).all(-1)
        mask = torch.logical_or(mask1, mask2)
        mask = torch.logical_not(mask)
        return seq[mask]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, seq = self.data[i]
        seq = torch.tensor(seq, dtype=torch.float32)
        img = img[0]
        if np.random.rand() > 0.5 and self.train:
            img = img[::-1]
            seq[:, 1] = -seq[:, 1]
            seq[:, 3] = -seq[:, 3]
        if np.random.rand() > 0.5 and self.train:
            img = img[:, ::-1]
            seq[:, 0] = -seq[:, 0]
            seq[:, 2] = -seq[:, 2]
        img = self.transform((img + 1) / 2)
        seq = ((seq + 1) / 2)
        if self.remove_artificial_edges:
            seq = self._remove_artificial_edges(seq)
        seq = self.sort_func(seq)
        return img, seq