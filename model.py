import torch
import torch.nn as nn
import torch.nn.functional as F


class simpleModel(nn.Module):
    def __init__(self, in_size, out_size):
        super(simpleModel, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.layer1 = nn.Sequential(
            nn.Linear(self.in_size, 3),
            nn.ReLU(),
        )
        self.clf = nn.Sequential(
            nn.Linear(3, self.out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.clf(out)
        return out