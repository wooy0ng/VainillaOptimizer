from load_dataset import *
from model import *
from torch.utils.data import DataLoader


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import pickle as pkl

def initialize_params(layers, batch):
    # initialize parameters
    _size_of_layers = len(layers) - 1
    params = {}

    for l in range(1, _size_of_layers+1):
        params['W' + str(l)] = torch.randn(batch, layers[l], layers[l-1]) * np.sqrt(1. / layers[l])
        params['b' + str(l)] = torch.randn(batch, layers[l]) * np.sqrt(1. / layers[l])

    return params

def criterion(outputs, labels):
    loss = torch.mean((outputs - labels) ** 2) / 2
    return loss

def train(args):
    print("\n[train mode]")
    dataset = load_dataset()

    batch = 1
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=True,
    )

    epochs = 50
    lr = 2.5e-2
    
    layers = [dataset.size(1), 3, 1]
    params = initialize_params(layers, batch)

    for epoch in range(epochs):
        losses = 0.
        _params = params.copy()
        for data, labels in data_loader:
            data = data.to(dtype=torch.float32)
            labels = labels.to(dtype=torch.float32)

            outputs, cache = forward(data, params)

            loss = criterion(outputs, labels)
            losses += loss.item()

            gradient = backward(outputs, labels, cache, params)
            _params = update(_params, gradient, lr, data.size(0))
        params = _params    # update parameters

        print(f"{epoch} epoch mean loss : {losses / len(data_loader):.3f}")
    pkl.dump(params, open('parameters.pkl', 'wb+'))


def test(args):
    print("\n[test mode]")
    dataset = load_dataset(mode=args.mode)
    
    batch = 1
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=False,
    )

    with open('parameters.pkl', 'rb+') as obj:
        params = pkl.load(obj)

    cnt = 0
    for data, labels in data_loader:
        data = data.to(dtype=torch.float32)
        labels = labels.to(dtype=torch.float32)

        outputs, _ = forward(data, params)
        
        predicted = (outputs > 0.5).float()
        predicted = predicted.squeeze().item()
        labels = labels.squeeze().item()
        if predicted == labels:
            cnt += 1
        print(f"predicted : {predicted},\tactual : {labels}")

    print(f"accuracy : {(cnt / len(data_loader)) * 100.:.3f}%")



    

    