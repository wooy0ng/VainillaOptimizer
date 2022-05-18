import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

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
        z = torch.matmul(W, prev_a[:, :, None]).squeeze(-1) + b    # W @ prev_a + b
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
    # da = outputs - labels >> MSE Loss
    da = F.binary_cross_entropy(outputs, labels.view(-1, 1))    # BCELoss
    loss = da.clone()

    for l in range(length, 0, -1):
        # backward
        prev_a = cache['a' + str(l-1)]
        z = cache['z' + str(l)]
        W = params['W' + str(l)]
        
        db = da * d_sigmoid(z)

        # error spot
        # dW = np.outer(db, prev_a)
        dW = torch.einsum('bi,bj->bij', (db, prev_a))
        da = torch.matmul(W.transpose(2, 1), db[:, :, None]).squeeze(-1)     # da = W.T @ db
        
        gradient['dW' + str(l)] = dW
        gradient['db' + str(l)] = db
    
    return gradient, loss
        
def update(params, gradient, learning_rate, m):
    length = len(params) // 2

    for l in range(1, length + 1):
        # gradient descent
        params['W' + str(l)] -= learning_rate * gradient['dW' + str(l)]
        params['b' + str(l)] -= learning_rate * gradient['db' + str(l)]
    return params