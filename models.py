import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.input_size = input_size

        self.fnn1 = nn.Sequential(
            nn.Linear(input_size,96),
            nn.ELU(),
            nn.BatchNorm1d(96),
            nn.Linear(96,1),
        )

    def forward(self, x):
        out = self.fnn1(x)
        return out