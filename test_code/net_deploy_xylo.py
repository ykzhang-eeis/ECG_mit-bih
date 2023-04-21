"""
    2023.4.11
    这段代码是很通用的，将训练好的网络部署到xylo硬件设备的模拟器上，返回的modSim就可以将其视为xylo硬件的网络，
    和net的数据类型是一致的，即可以使用modSim(data)来做后续处理，但是目前还没有写相应的net，留待后续处理
"""


# - Rockpool time-series handling
from rockpool import TSEvent, TSContinuous
import torch
from rockpool.nn.modules import LinearTorch, LIFTorch
from rockpool.parameters import Constant
from rockpool.nn.combinators import Sequential
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
from rockpool.transform import quantize_methods as q
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm.asyncio import tqdm
from rockpool.nn.networks.wavesense import WaveSenseNet
from rockpool.devices import xylo as x

g = net.as_graph()
spec = x.vA2.mapper(g, weight_dtype='float', threshold_dtype='float', dash_dtype='float')
quant_spec = spec.copy()
# - Quantize the specification
spec.update(q.global_quantize(**spec))
# - Use rockpool.devices.xylo.config_from_specification
config, is_valid, msg = x.vA2.config_from_specification(**spec)
modSim = x.vA2.XyloSim.from_config(config)