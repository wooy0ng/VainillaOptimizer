from load_dataset import *
from model import *
from torch.utils.data import DataLoader


import torch.optim as optim
import torch.nn as nn

def initialize_params(layers):
    # initialize parameters
    _size_of_layers = len(layers) - 1
    params = {}

    for l in range(1, _size_of_layers+1):
        params['W' + str(l)] = np.random.randn(layers[l], layers[l-1]) * np.sqrt(1. / layers[l])
        params['b' + str(l)] = np.random.randn(layers[l]) * np.sqrt(1. / layers[l])

    return params

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(x, params):
    "forward function"
    cache = {'a0': x}
    length = len(params) // 2

    for l in range(1, length+1):
        prev_a = cache['a' + str(l-1)]
        W = params['W' + str(l)]
        b = params['b' + str(l)]

        # calculate z, a
        z = W @ prev_a + b
        a = sigmoid(z)

        # save to cache
        cache['z' + str(l)] = z
        cache['a' + str(l)] = a

    return a, cache
        

def train():
    dataset = load_dataset()

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
    )

    epochs = 30
    lr = 1e-2

    layers = [dataset.size(0), 3, 2]
    params = initialize_params(layers)

    for epoch in range(epochs):
        _params = params.copy()
        for data, labels in data_loader:
            data = data.numpy()
            labels = labels.numpy()
            outputs, cache = forward(data, _params)