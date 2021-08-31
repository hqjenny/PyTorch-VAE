from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
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
        print(f"Path to CSV file: {csv_file}")
        full_dataset = shuffle(pd.read_csv(csv_file), random_state=0)
        self.transform = transform
        self.target_transform = target_transform

        num_rows = len(full_dataset)
        train_part = int (0.75 * num_rows)
        test_part = int (0.15 * num_rows)
        if split == "train":
            self.arch_feats_frame = full_dataset[0: train_part]
        elif split == "valid":
            self.arch_feats_frame = full_dataset[train_part:-test_part]
        elif split == "test":
            self.arch_feats_frame = full_dataset[-test_part:]
        else:
            self.arch_feats_frame = full_dataset
        dataset_size = len(self.arch_feats_frame)
        print(f"Load dataset with {dataset_size} entries.")
        
    def __len__(self):
        return len(self.arch_feats_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # arch_feats = self.arch_feats_frame.iloc[idx, 3:]
        # arch_feats = self.arch_feats_frame.iloc[idx, [3,4,5,7,8,9,10,11,12,14]]
        arch_feats = self.arch_feats_frame.iloc[idx, [3, 4, 5, 8, 10, 12, 14]]

        label = self.arch_feats_frame.iloc[idx, 1]

        arch_feats = np.array([arch_feats])
        arch_feats = arch_feats.astype('double').reshape(-1, ).astype(np.float64)
    
        if self.transform:
            arch_feats = self.transform(arch_feats)
        if self.target_transform:
            label = self.target_transform(label)

        return arch_feats, label

if __name__ == "__main__":
    train_dataset = CoSADataset(root = './', split= "train", )
    test_dataset = CoSADataset(root = './', split= "test",)
