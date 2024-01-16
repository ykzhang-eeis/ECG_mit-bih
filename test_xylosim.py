import torch
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from rockpool.devices.xylo.syns61201 import mapper, config_from_specification
from rockpool.devices.xylo.syns61201 import XyloSim
from rockpool.transform import quantize_methods as q
from rockpool.timeseries import TSEvent

try:
    from rich import print
except:
    pass

from model import MyNet

warnings.filterwarnings('ignore')

model = MyNet
state_dict = torch.load("output/model_weights.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

spec = mapper(model.as_graph(), weight_dtype="float")
quant_spec = spec.copy()
del quant_spec["mapped_graph"]
del quant_spec["dt"]
quant_spec.update(q.global_quantize(**quant_spec))
config, is_valid, msg = config_from_specification(**quant_spec)

modSim = XyloSim.from_config(config=config, dt=1e-3)

predictions, targets = [], []
def xyloSim_inference(test_dataloader, mod=modSim):

    for batch, target in test_dataloader:
        # mod.reset_state()
        # mod.reset_parameters()
        output, _, rec_sim = mod(batch.squeeze(0).cpu().numpy(), record=True)
        # - Plot output and target
        peaks = output.sum(0)
        predictions.append(peaks.argmax(0))
        targets.append(target.cpu().numpy().item())

    return accuracy_score(targets, predictions)