from load_dataset import *
from model import *
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn


def train():
    dataset = load_dataset()

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
    )

    epochs = 30
    lr = 1e-2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = simpleModel(12, 1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        losses = 0.
        for data, label in data_loader:
            data = data.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.float32)

            outputs = model(data)
            loss = criterion(outputs, label.view(-1, 1))
            losses += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"loss : {losses / len(data_loader)}")