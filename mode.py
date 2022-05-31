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
    '''
    # MSELoss
    loss = torch.mean((outputs - labels) ** 2) / 2
    '''
    # cross Entropy loss
    labels = torch.argmax(labels, dim=1)
    loss = F.cross_entropy(outputs, labels)
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
    lr = 0.01
    
    layers = [dataset.size(1), 3, 2]
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
            _params = step(_params, gradient, lr, data.size(0))
        params = _params    # update parameters

        print(f"{epoch} epoch mean loss : {losses / len(data_loader):.3f}")
    pkl.dump(params, open('parameters.pkl', 'wb+'))
    pkl.dump(dataset, open('dataset.pkl', 'wb+'))


def test(args):
    print("\n[test mode]")
    args.mode = 'test'
    # dataset = load_dataset(mode=args.mode)
    
    with open('parameters.pkl', 'rb+') as obj:
        params = pkl.load(obj)
    
    with open('dataset.pkl', 'rb+') as obj:
        dataset = pkl.load(obj)
        dataset(args.mode)
    
    
    batch = 1
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=False,
    )

    

    cnt = 0
    for data, labels in data_loader:
        data = data.to(dtype=torch.float32)
        labels = labels.to(dtype=torch.float32)

        outputs, _ = forward(data, params)
        
        predicted = torch.argmax(outputs, dim=1).item()
        labels = torch.argmax(labels, dim=1).item()
        if predicted == labels:
            cnt += 1
        print(f"predicted : {predicted},\tactual : {labels}")

    print(f"accuracy : {(cnt / len(data_loader)) * 100.:.3f}%")

    # visualized image
    iterator = iter(data_loader)
    inputs, labels = next(iterator)
    
    outputs, _ = forward(data, params)

    predicted = torch.argmax(outputs, dim=1).item()
    labels = torch.argmax(labels, dim=1).item()
    inputs = inputs.view(-1, 4, 3)
    visualized_image(inputs, title=f"predicted : {predicted}  actual : {labels}")



    

    