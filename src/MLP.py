import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import warnings
import torchtext
import math


def GELU(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x**3)))


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, hidden_act, criterion, optimizer):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(in_dim, hid_dim)
        if hidden_act == "GeLU":
            self.hidden_act = nn.GELU()
        else:
            self.hidden_act = nn.ReLU()

        self.output_layer = nn.Linear(hid_dim, out_dim)
        self.criterion = criterion
        self.optimizer = optimizer(self.parameters(), lr=0.01)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_act(x)
        x = self.output_layer(x)
        return x

    def backward(self, y_target):
        gradients = {p: torch.zeros_like(p.data) for p in self.parameters()}

        y_pred = self(x)
        self.loss = self.criterion(y_pred, y_target)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        print(y_pred)
        print(y_target)
        for p in self.parameters():
            gradients[p] += p.grad.data

        for p in self.parameters():
            p.grad.zero_()

        return gradients


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    batch_size = 64

    mnist_trainset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    mnist_testset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        mnist_trainset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        mnist_testset, batch_size=batch_size, shuffle=False
    )

    images, labels = next(iter(train_loader))

    print(f"Shape of image: {images[0].size()}")

    # model = MLP(1, 96, 1, "ReLU", nn.MSELoss(), torch.optim.SGD)
    #
    # # Define the loss function
    # # criterion = nn.MSELoss()
    #
    # # Define the optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    #
    # # Generate some example data
    # x = torch.randn(100, 1)
    # y = x.pow(2) + 0.2 * torch.randn(100, 1)
    #
    # # Train the model
    # for epoch in range(1000):
    #     # Forward pass
    #     y_pred = model(x)
    #     # loss = criterion(y_pred, y)
    #
    #     # Backward pass and optimization
    #     model.backward(y)
    #     # for p in model.parameters():
    #     # p.data -= 0.01 * gradients[p]
    #
    #     if epoch % 100 == 0:
    #         print("Epoch {}: loss={}".format(epoch, model.loss.item()))
