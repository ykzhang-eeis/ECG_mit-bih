import torch
from rockpool.devices import xylo as x
from rockpool.transform import quantize_methods as q
from sklearn.metrics import accuracy_score
from model import My_net
# - Pretty printing
try:
    from rich import print
except:
    pass

# - Disable warnings
import warnings
warnings.filterwarnings('ignore')

model = My_net
model.eval()
state_dict = torch.load("output/model_weights.pth", map_location="cpu")
model.load_state_dict(state_dict)

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

def xylo_inference(test_dataloader, model=modSamna):

    predictions, targets = [], []

    for batch, target in test_dataloader:
        batch = batch.cpu().numpy()
        target = target.type(torch.long)
        model.reset_state() # 模型维护了一种内部状态，这种状态如果不重置，会对模型的预测产生负面影响
        out, _, recordings = model(batch, record=True)
        predictions.extend(out.argmax(1))
        targets.extend(target.cpu().numpy())

    accuracy = accuracy_score(targets, predictions)
    print(f'Infer accuracy: {accuracy}')