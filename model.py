from rockpool.nn.modules import LIFBitshiftTorch
from rockpool.nn.modules import LinearTorch
from rockpool.nn.combinators import Sequential, Residual
from rockpool.parameters import Constant
from rockpool.nn.modules.torch.lif_torch import PeriodicExponential

Nin = 15
Nout = 4
dt = 1e-3
Nhidden = 63

My_net = Sequential(
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