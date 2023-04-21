# - Numpy
import numpy as np

# - Matplotlib
import sys
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams['figure.dpi'] = 300

# - Rockpool time-series handling
from rockpool import TSEvent, TSContinuous

# - Pretty printing
try:
    from rich import print
except:
    pass

# - Display images
from IPython.display import Image

# - Disable warnings
import warnings
warnings.filterwarnings('ignore')

# - Import the computational modules and combinators required for the networl
from rockpool.nn.modules import LIFTorch, LinearTorch
from rockpool.nn.combinators import Sequential, Residual
from rockpool.devices import xylo as x
from rockpool.transform import quantize_methods as q

# - Define the size of the network layers
Nin = 2
Nhidden = 4
Nout = 2
dt = 1e-3

# - Define the network architecture using combinators and modules
net = Sequential(
    LinearTorch((Nin, Nhidden), has_bias = False),
    LIFTorch(Nhidden, dt = dt),
    
    Residual(
        LinearTorch((Nhidden, Nhidden), has_bias = False),
        LIFTorch(Nhidden, has_rec = True, threshold = 10., dt = dt),
    ),
    
    LinearTorch((Nhidden, Nout), has_bias = False),
    LIFTorch(Nout, dt = dt),
)
print(net)

# - Scale down recurrent weights for stability
net[2][1].w_rec.data = net[2][1].w_rec / 10.

g = net.as_graph()
spec = x.vA2.mapper(g, weight_dtype='float', threshold_dtype='float', dash_dtype='float')
print(spec)
quant_spec = spec.copy()
# - Quantize the specification
spec.update(q.global_quantize(**spec))
print(spec)
# - Use rockpool.devices.xylo.config_from_specification
config, is_valid, msg = x.vA2.config_from_specification(**spec)
modSim = x.vA2.XyloSim.from_config(config)
print(modSim)

T = 100
f = 0.1
input_spikes = np.random.rand(T, Nin) < f
TSEvent.from_raster(input_spikes, dt, name = 'Poisson input events').plot();
out, _, r_d = modSim(input_spikes, record = True)
print(r_d.keys())
# - Plot some internal state variables
plt.figure()
plt.imshow(r_d['Spikes'].T, aspect = 'auto', origin = 'lower')
plt.title('Hidden spikes')
plt.ylabel('Channel')

plt.figure()
TSContinuous.from_clocked(r_d['Vmem'], dt, name = 'Hidden membrane potentials').plot(stagger = 127)

plt.figure()
TSContinuous.from_clocked(r_d['Isyn'], dt, name = 'Hidden synaptic currents').plot(stagger = 127);
plt.show()