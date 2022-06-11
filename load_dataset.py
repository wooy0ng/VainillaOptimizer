import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random

from torch.utils.data import Dataset

class load_dataset(Dataset):
    def __init__(self, mode='train'):
        super(load_dataset, self).__init__()

        df = pd.read_excel('./dataset.xlsx', sheet_name='Data')
        
        idx = 10
        tmp_arrs, tmp_labels = [], []
        while idx < df.shape[-1]:
            tmp_df = df.iloc[3:7, idx:idx+3]; tmp_label = df.iloc[7, idx+3]
            tmp_arrs.append(tmp_df.to_numpy().flatten())
            tmp_labels.append(int(tmp_label))
            idx += 4
        
        arrs = np.asarray(tmp_arrs)
        labels = np.asarray(pd.get_dummies(tmp_labels))
        
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
            arrs, labels, test_size=0.33, random_state=45
        )

        if mode == 'train':
            self._arrs = self.train_X
            self._labels = self.train_y
        else:
            self._arrs = self.test_X
            self._labels = self.test_y
    
    def __call__(self, mode):
        if mode == 'train':
            self._arrs = self.train_X
            self._labels = self.train_y
        else:
            self._arrs = self.test_X
            self._labels = self.test_y
    
    def __len__(self):
        return len(self._arrs)

    def __getitem__(self, idx):
        arr = self._arrs[idx]
        label = self._labels[idx]
        return torch.tensor(arr), torch.tensor(label)

    def size(self, idx):
        return self._arrs.shape[idx]

    