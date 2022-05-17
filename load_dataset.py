import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

class load_dataset(Dataset):
    def __init__(self):
        super(load_dataset, self).__init__()

        df = pd.read_excel('./dataset.xlsx')
        labels_df = df['label']
        df.drop(['label'], axis=1, inplace=True)
        
        # self.arrs = np.asarray(df).reshape(-1, 4, 3)
        self.arrs = np.asarray(df)
        self.labels = np.asarray(labels_df)

    
    def __len__(self):
        return len(self.arrs)

    def __getitem__(self, idx):
        arr = self.arrs[idx]
        label = np.asarray(self.labels[idx])

        return torch.tensor(arr), torch.tensor(label)


    