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

    def __init__(self, root='.', split='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        csv_file = f'{root}/arch_dataset_12.csv'
        print(csv_file)
        self.landmarks_frame = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # print(self.landmarks_frame.iloc[idx, 0])
        landmarks = self.landmarks_frame.iloc[idx, 3:]
        label = self.landmarks_frame.iloc[idx, 1]
        print(f'cycle: {label}')
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, ).astype(np.float64)
        sample = {'image': landmarks, 'label': label}
        # print(f'sample: {sample}')

        return landmarks, 1
