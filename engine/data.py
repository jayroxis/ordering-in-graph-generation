from typing import Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from utils.data.dataset import *
from utils.data.misc import PadSequence


class DataModule(pl.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.data_config = data_config
        self.dataset = None
        self.train_loader = None

    def setup(self, stage: Optional[str] = None):
        latent_sort_encoder = self.data_config.get("latent_sort_encoder")
        if latent_sort_encoder is not None:
            print("[INFO] Using Latent Ordering.")
            self.dataset = LatentSortRejectConfusionDataset(
                encoder=torch.jit.load(latent_sort_encoder),
                **self.data_config
            )
        else:
            self.dataset = RenderedPlanarGraphDataset(**self.data_config)

        train_size = int(len(self.dataset) * 0.8)
        valid_size = len(self.dataset) - train_size
        self.train_dataset, self.valid_dataset = random_split(self.dataset, [train_size, valid_size])

        pad_value = self.data_config['pad_value']
        self.train_loader = DataLoader(
            self.train_dataset,
            collate_fn=PadSequence(pad_value),
            batch_size=self.data_config['batch_size'],
            shuffle=self.data_config['shuffle'],
            num_workers=self.data_config['num_workers'],
            pin_memory=self.data_config['pin_memory']
        )

    def train_dataloader(self):
        return self.train_loader


