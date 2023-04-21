import torch
import numpy as np
import sys
import matplotlib.pyplot as plt

from torch.nn import MSELoss
from torch.optim import Adam

from tqdm.autonotebook import tqdm

from rockpool import TSEvent
from rockpool.parameters import Constant
from rockpool.nn.modules import LIFBitshiftTorch, LinearTorch
from rockpool.nn.combinators import Sequential, Residual
from rockpool.nn.modules.torch.lif_torch import PeriodicExponential

try:
    from rich import print
except:
    pass

plt.rcParams["figure.figsize"] = [12, 4]
plt.rcParams["figure.dpi"] = 300

Nin = 16
Nout = 4
num_classes = 4
T = 50
dt = 1e-3
Nhidden = 63
net = Sequential(
    LinearTorch((Nin, Nhidden), has_bias=False),
    LIFBitshiftTorch(
        (Nhidden),
        tau_mem=0.002,
        tau_syn=0.002,
        bias=Constant(0),
        dt=dt,
        spike_generation_fn=PeriodicExponential,
    ),
    Residual(
        LinearTorch((Nhidden, Nhidden), has_bias=False),
        LIFBitshiftTorch(
            (Nhidden),
            tau_mem=0.002,
            tau_syn=0.002,
            bias=Constant(0),
            dt=dt,
            spike_generation_fn=PeriodicExponential,
        ),
    ),
    LinearTorch((Nhidden, Nout), has_bias=False),
    LIFBitshiftTorch(
        (Nout),
        tau_mem=0.002,
        tau_syn=0.002,
        bias=Constant(0),
        dt=dt,
        spike_generation_fn=PeriodicExponential,
    ),
)

optimizer = Adam(net.parameters().astorch(), lr=1e-2)

loss_fun = MSELoss()

# - Record the loss values over training iterations
loss_t = []

num_epochs = 1000
iterator = tqdm(range(num_epochs))

for i in iterator:
    epoch_loss = 0
    for input, target in ds:

        net.reset_state()
        optimizer.zero_grad()

        output, _, record = net(input.unsqueeze(0), record=True)

        loss = loss_fun(output, target.unsqueeze(0))

        if loss.item() > 0:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

    # - Keep track of the loss
    loss_t.append(epoch_loss)