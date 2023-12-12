from model.my_model import *
from rockpool.devices import xylo as x
from rockpool.transform import quantize_methods as q

from model.my_model import My_net
# - Pretty printing
try:
    from rich import print
except:
    pass

# - Disable warnings
import warnings
warnings.filterwarnings('ignore')

model = My_net

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

if found_xylo:
    modSamna = x.XyloSamna(hdk, config, dt = 0.01)
    print(modSamna)