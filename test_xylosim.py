import torch
import numpy as np
import warnings
from sklearn.metrics import accuracy_score
from rockpool.devices.xylo.syns61201 import mapper, config_from_specification
from rockpool.devices.xylo.syns61201 import XyloSim
from rockpool.transform import quantize_methods as q

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
spec.update(q.global_quantize(**spec))
config, is_valid, msg = config_from_specification(**spec)

modSim = XyloSim.from_config(config=config)
print(modSim)

def xyloSim_inference(test_dataloader, mod=modSim):
    predictions, targets = [], []

    for batch, target in test_dataloader:
        batch = batch.squeeze(0).cpu().numpy().astype(int)
        target = target.type(torch.long)
        mod.reset_state() # 模型维护了一种内部状态，这种状态如果不重置，会对模型的预测产生负面影响
        out, _, recordings = mod(batch, record=True)
        peaks = np.sum(out, axis=0)
        predictions.append(peaks.argmax())
        targets.append(target.cpu().numpy())

    accuracy = accuracy_score(targets, predictions)
    print(f'Infer accuracy: {accuracy}')