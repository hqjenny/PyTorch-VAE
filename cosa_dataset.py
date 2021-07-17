from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class CoSADataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root='.', split='train', transform=None, target_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        csv_file = f'{root}/arch_dataset_12.csv'
        print(csv_file)
        self.arch_feats_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.arch_feats_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        arch_feats = self.arch_feats_frame.iloc[idx, 3:]
        label = self.arch_feats_frame.iloc[idx, 1]
        print(f'cycle: {label}')
        arch_feats = np.array([arch_feats])
        arch_feats = arch_feats.astype('float').reshape(-1, ).astype(np.float64)
    
        if self.transform:
            arch_feats = self.transform(arch_feats)
        if self.target_transform:
            label = self.target_transform(label)

        return arch_feats, label
