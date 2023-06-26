import torch
from model.my_model import *
from rockpool.devices import xylo as x
from rockpool.transform import quantize_methods as q
import torch
import numpy as np
import wfdb
import pywt
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

# - Import the computational modules and combinators required for the network
from rockpool import TSEvent, TSContinuous
from rockpool.nn.modules import LinearTorch, LIFTorch
from rockpool.parameters import Constant
from rockpool.nn.networks import SynNet,WaveSenseNet
from rockpool.nn.combinators import Sequential, Residual
from rockpool.transform import quantize_methods as q
from pathlib import Path
from rockpool.devices import xylo as x
from data_process import *
from encode import sigma_delta_encoding, BSA_encoding, BSA_decoding
from params import *
from dataloader import *
from model_train import model_train
from model.my_model import My_net
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

model = My_net
# X_data, Y_data = loadData()
# print(X_data.shape, Y_data.shape)
# print("data ok!!!")
# X_data = np.reshape(X_data, (-1, 300)) # 将shape从(286892,300,1)转变为(286892,300)
# print(X_data.shape, Y_data.shape)
# dataset = ECG_Dataset(X_data, Y_data)
# train_size = int((1-dataset_params["RATIO"]) * len(dataset))
# test_size = int(dataset_params["RATIO"] * len(dataset))
# val_size = len(dataset) - train_size - test_size
# train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
# # for (inputs, targets) in dataset:
# #     print(f"输入的sigma-delta编码矩阵为\n{inputs}\n对应的类别为第{targets}类")
# train_dataloader = DataLoader(train_dataset, batch_size=training_params["Batch_Size"], shuffle=True, num_workers=16)
# val_dataloader = DataLoader(val_dataset, batch_size=training_params["Batch_Size"], num_workers=16)
# test_dataloader = DataLoader(test_dataset, batch_size=training_params["Batch_Size"], num_workers=16)
# state_dict = torch.load("output/model_weights.pth")
# model.load_state_dict(state_dict)
# g = model.as_graph()
# spec = x.vA2.mapper(g, weight_dtype='float', threshold_dtype='float', dash_dtype='float')
# quant_spec = spec.copy()
# # - Quantize the specification
# spec.update(q.global_quantize(**spec))
# config, is_valid, msg = x.vA2.config_from_specification(**spec)
# print(is_valid)
# modSim = x.vA2.XyloSim.from_config(config)
# test_iterator = iter(test_dataloader)
# batch, target = next(test_iterator)
# for i in range(batch.shape[0]):
#     modSim_output, _, r_d = modSim(batch[i].numpy(), record = True)
#     modSim_peaks = modSim_output.max(1)[0]
#     print(target[i]==modSim_peaks)


from rockpool.devices.xylo import find_xylo_hdks
# from rockpool.devices.xylo import find_xylo_hdks, xylo_devkit_utils
# xylo_hdk_nodes = xylo_devkit_utils.find_xylo_boards(samna.device_node)
# - Detect a connected HDK and import the required support package
connected_hdks, support_modules, chip_versions = find_xylo_hdks()
found_xylo = len(connected_hdks) > 0
if found_xylo:
    hdk = connected_hdks[0]
    x = support_modules[0]
else:
    assert False, 'This tutorial requires a connected Xylo HDK to run.'
spec = x.mapper(model.as_graph(), weight_dtype = 'float')
spec.update(q.global_quantize(**spec))
# - Use rockpool.devices.xylo.config_from_specification
config, is_valid, msg = x.config_from_specification(**spec)
# - Use rockpool.devices.xylo.XyloSamna to deploy to the HDK
if found_xylo:
    modSamna = x.XyloSamna(hdk, config, dt = 0.01)
    print(modSamna)