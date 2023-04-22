import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import torchtext


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(in_dim, hid_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hid_dim, out_dim)

    def _forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    point_MLP = MLP(3, 96, 96)

    x = torch.ones((10, 3))

    proc_x = point_MLP._forward(x)

    print(proc_x)
