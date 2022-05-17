from load_dataset import *
from model import *
from torch.utils.data import DataLoader


import torch.optim as optim
import torch.nn as nn

def initialize_params(layers, batch):
    # initialize parameters
    _size_of_layers = len(layers) - 1
    params = {}

    for l in range(1, _size_of_layers+1):
        params['W' + str(l)] = np.random.randn(batch, layers[l], layers[l-1]) * np.sqrt(1. / layers[l])
        params['b' + str(l)] = np.random.randn(batch, layers[l]) * np.sqrt(1. / layers[l])

    return params

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return (np.exp(-1)) / ((np.exp(-x) + 1)**2)

def forward(x, params):
    ''' forward function '''
    cache = {'a0': x}
    length = len(params) // 2

    for l in range(1, length+1):
        prev_a = cache['a' + str(l-1)]
        W = params['W' + str(l)]
        b = params['b' + str(l)]

        # calculate z, a
        z = np.matmul(W, prev_a[:, :, None]).squeeze(-1) + b    # W @ prev_a + b
        a = sigmoid(z)

        # save to cache
        cache['z' + str(l)] = z
        cache['a' + str(l)] = a

    return a, cache

def backward(outputs, labels, cache, params):
    ''' backward function '''
    gradient = {}
    length = len(cache) // 2
    
    # criterion
    da = outputs - labels
    
    
    for l in range(length, 0, -1):
        # backward
        prev_a = cache['a' + str(l-1)]
        z = cache['z' + str(l)]
        W = params['W' + str(l)]
        
        db = da * d_sigmoid(z)
        dW = np.outer(db, prev_a)
        da = np.matmul(W.transpose(0, 2, 1), db[:, :, None]).squeeze(-1)     # da = W.T @ db
        
        gradient['dW' + str(l)] = dW
        gradient['db' + str(l)] = db
    
    return gradient
        
def update(params, gradient, learning_rate):
    length = len(params) // 2

    for l in range(1, length + 1):
        # gradient descent
        params['W' + str(l)] -= learning_rate * gradient['dW' + str(l)]
        params['b' + str(l)] -= learning_rate * gradient['db' + str(l)]
    
    return params


def train():
    dataset = load_dataset()

    batch = 4
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=True,
    )

    epochs = 30
    lr = 1e-2
    
    layers = [dataset.size(1), 3, 2]
    params = initialize_params(layers, batch)

    for epoch in range(epochs):
        _params = params.copy()
        for data, labels in data_loader:
            data = data.numpy()
            labels = labels.numpy()

            outputs, cache = forward(data, params)
            gradient = backward(outputs, labels, cache, params)
            _params = update(_params, gradient, lr)