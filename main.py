import torch
import numpy as np
import wfdb
import pywt
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

from torchsummary import summary
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
from model_train import model_train_ann, model_train_snn
from model.my_model import My_net
from model.ann_model import ann_net
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


def main():
    X_data, Y_data = loadData()
    print(X_data.shape, Y_data.shape)
    print("data ok!!!")
    X_data = np.reshape(X_data, (-1, 300)) # 将shape从(286892,300,1)转变为(286892,300)
    print(X_data.shape, Y_data.shape)
    dataset = ECG_Dataset(X_data, Y_data)
    train_size = int((1-dataset_params["RATIO"]) * len(dataset))
    test_size = int(dataset_params["RATIO"] * len(dataset))
    val_size = len(dataset) - train_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=training_params["Batch_Size"], shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=training_params["Batch_Size"], num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=training_params["Batch_Size"], shuffle=True, num_workers=0)
    # model_train(train_dataloader, test_dataloader, model)
    # model_train(train_dataloader, test_dataloader, SynNet(n_classes=dataset_params["CLASSES"],n_channels=dataset_params["Time_Partitions"]))
    # model_train(train_dataloader, test_dataloader, SynNet(n_classes=dataset_params["CLASSES"],n_channels=300))
    # model_train(train_dataloader, test_dataloader, SynNet(n_classes=dataset_params["CLASSES"],n_channels=2))
    # model_train(train_dataloader, test_dataloader, WaveSenseNet(dilations=[2, 32],n_classes=dataset_params["CLASSES"],n_channels_in=1))
    model_train_snn(train_dataloader, test_dataloader,My_net)

'''
def main():
    model = ann_net.to(device="cuda")
    summary(model, input_size=[(30,2)], batch_size=16, device="cuda")
    state_dict = torch.load("output/model_weights.pth",map_location=torch.device('cuda'))
    model.load_state_dict(state_dict)
    X_data, Y_data = loadData()
    print(X_data.shape, Y_data.shape)
    print("data ok!!!")
    X_data = np.reshape(X_data, (-1, 300)) # 将shape从(286892,300,1)转变为(286892,300)
    print(X_data.shape, Y_data.shape)
    dataset = ECG_Dataset(X_data, Y_data)
    for (inputs, targets) in dataset:
        preds = model(inputs.unsqueeze(0).to("cuda").to(torch.float32)).argmax(1)
        print(f"{preds == targets}")
'''

if __name__ == '__main__':
    main()