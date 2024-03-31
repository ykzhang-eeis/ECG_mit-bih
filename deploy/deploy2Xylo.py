import torch
import warnings
from sklearn.metrics import accuracy_score
from rockpool.devices.xylo import find_xylo_hdks
from rockpool.transform import quantize_methods as q

# - Pretty printing
try:
    from rich import print
except:
    pass

from models import MyNet

warnings.filterwarnings('ignore')

model = MyNet
state_dict = torch.load("output/model_weights.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

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

mod_samna = x.XyloSamna(hdk, config, dt = 0.001)

data = torch.rand((16, 2, 15))
output = model(data)

def xylo_inference(test_dataloader, mod_samna=mod_samna):
    predictions, targets = [], []

    for batch, target in test_dataloader:
        batch = batch.cpu().numpy().astype(int)
        target = target.type(torch.long)
        mod_samna.reset_state() # The model maintains an internal state, which will have a negative impact on the model's predictions if not reset
        out, _, recordings = mod_samna(batch, record=True)
        predictions.extend(out.argmax(1))
        targets.extend(target.cpu().numpy())

    accuracy = accuracy_score(targets, predictions)
    print(f'Infer accuracy: {accuracy}')