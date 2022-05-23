import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return torch.sigmoid(x)

def d_sigmoid(x):
    return torch.sigmoid(x) * (1 - torch.sigmoid(x))

def softmax(x):
    return torch.softmax(x, dim=1)

def d_softmax(x):
    return torch.softmax(x, dim=1) - x

def visualized_image(img, title):
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.title(title)
    plt.savefig("./test.png")
    plt.clf()

