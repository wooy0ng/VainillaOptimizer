from load_dataset import *
from model import *
from torch.utils.data import DataLoader


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import pickle as pkl

def initialize_params(layers):
    # initialize parameters
    _size_of_layers = len(layers) - 1
    params = {}

    for l in range(1, _size_of_layers+1):
        params['W' + str(l)] = torch.tensor(np.random.randn(layers[l], layers[l-1]) * np.sqrt(1. / layers[l])).to(dtype=torch.float32)
        params['b' + str(l)] = torch.tensor(np.random.randn(layers[l]) * np.sqrt(1. / layers[l])).to(dtype=torch.float32)

    return params

def criterion(outputs, labels):
    '''
    # Cross Entropy Loss Function
    mse loss : torch.mean((outputs - labels) ** 2) / 2
    cross_entropy : F.cross_entropy(outputs, labels)
    '''
    # cross Entropy loss
    log_softmax = torch.log(outputs)
    loss = (labels * (-log_softmax)).sum(dim=0).mean()
    return loss

def train(args):
    print("\n[train mode]")
    dataset = load_dataset()

    epochs = 50
    lr = 0.5
    
    layers = [dataset.size(1), 3, 2]
    params = initialize_params(layers)

    for epoch in range(epochs):
        losses, count = 0., 0
        _params = params.copy()
        for data, labels in dataset:
            data = data.to(dtype=torch.float32)
            labels = labels.to(dtype=torch.float32)

            outputs, cache = forward(data, params)
            loss = criterion(outputs, labels)
            losses += loss.item()

            gradient = backward(outputs, labels, cache, params)
            _params = step(_params, gradient, lr, dataset.size(0))
            count += 1
        params = _params    # update parameters

        print(f"{epoch + 1} epoch mean loss : {losses / count:.3f}")
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
    
    cnt = 0
    for data, labels in dataset:
        data = data.to(dtype=torch.float32)
        labels = labels.to(dtype=torch.float32)

        outputs, _ = forward(data, params)
        
        predicted = torch.argmax(outputs, dim=0).item()
        labels = torch.argmax(labels, dim=0).item()
        if predicted == labels:
            cnt += 1
        print(f"predicted : {predicted},\tactual : {labels}")

    print(f"accuracy : {(cnt / dataset.size(0)) * 100.:.3f}%")

    # visualized image
    inputs_list, labels_list, titles = [], [], []
    for data, labels in dataset:
        data = data.to(dtype=torch.float32)
        labels = labels.to(dtype=torch.float32)
        
        outputs, _ = forward(data, params)
        predicted = torch.argmax(outputs, dim=0).item()
        labels = torch.argmax(labels, dim=0).item()    
        
        inputs_list.append(data.view(-1, 4, 3))
        labels_list.append(labels)
        titles.append(f"{predicted} / {labels}")
    
    test_list = list(zip(inputs_list, titles))
    random.shuffle(test_list)
    
    img_cnt = 5
    fig, ax = plt.subplots(1, img_cnt)
    for idx ,(data, title) in enumerate(test_list):
        if idx >= img_cnt:
            break
        img = data.numpy().transpose((1, 2, 0))
        ax[idx].imshow(img, cmap=matplotlib.colors.ListedColormap(['white', 'black']))
        ax[idx].set_title(title)
        ax[idx].set_xticks([])
        ax[idx].set_yticks([])

    
    fig.suptitle('predicted / actual')
    plt.savefig('./test.png', dpi=200)
    plt.clf()
        
    

    