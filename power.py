import numpy as np
import torch
from rockpool.devices import xylo as x
from rockpool.transform import quantize_methods as q
from model import MyNet
try:
    from rich import print
except:
    pass

model = MyNet
state_dict = torch.load("output/model_weights.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

from rockpool.devices.xylo import find_xylo_hdks

connected_hdks, support_modules, chip_versions = find_xylo_hdks()
found_xylo = len(connected_hdks) > 0

if found_xylo:
    hdk = connected_hdks[0]
    x = support_modules[0]
else:
    assert False, 'This tutorial requires a connected Xylo HDK to run.'

spec = x.mapper(model.as_graph(), weight_dtype = 'float')
spec.update(q.global_quantize(**spec))
config, is_valid, msg = x.config_from_specification(**spec)

# - Use rockpool.devices.xylo.XyloSamna to deploy to the HDK
modSamna = x.XyloSamna(hdk, config, dt = 0.01)

io_power_list = []
logic_power_list = []

data = torch.rand((500, 123), dtype=torch.float)
data = data.numpy().astype(int)

modSamna.reset_state()

out, _, recordings = modSamna((data*3).clip(0, 15),record=False,record_power = True,read_timeout = 40)
print(recordings)

io_power = np.mean(recordings['io_power'])
logic_power = np.mean(recordings['logic_power'])
print('io_power:',io_power,'logic_power:',logic_power)

# io_power_list.append(io_power)
# logic_power_list.append(logic_power)