import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

class load_dataset(Dataset):
    def __init__(self, mode='train'):
        super(load_dataset, self).__init__()

        df = pd.read_excel('./dataset.xlsx')
        labels_df = df['label']
        df.drop(['label'], axis=1, inplace=True)
        
        arrs = np.asarray(df)
        labels = np.asarray(labels_df)

        train_X, test_X, train_y, test_y = train_test_split(
            arrs, 
            labels, 
            test_size=0.2, 
            random_state=42
        )

        if mode == 'train':
            self._arrs = train_X
            self._labels = train_y
        else:
            self._arrs = test_X
            self._labels = test_y
    
    def __len__(self):
        return len(self._arrs)

    def __getitem__(self, idx):
        arr = self._arrs[idx]
        label = self._labels[idx]

        return torch.tensor(arr), torch.tensor(label)

    def size(self, idx):
        return self._arrs.shape[idx]
    