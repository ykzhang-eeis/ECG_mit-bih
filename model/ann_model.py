from params import *
import torch.nn as nn

Nin = 2
Nout = 4
Nhidden = 63

ann_net = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(30*32, 4),
    nn.Softmax(dim=1)
)