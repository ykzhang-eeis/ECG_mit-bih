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

def main():
    # X_train,Y_train为所有的数据集和标签集
    # X_test,Y_test为拆分的测试集和标签集
    X_data, Y_data = loadData()
    print(X_data.shape, Y_data.shape)
    print("data ok!!!")
    X_data = np.reshape(X_data, (-1, 300)) # 将shape从(286892,300,1)转变为(286892,300)
    print(X_data.shape, Y_data.shape)
    # pd.DataFrame(X_train).to_csv('X_train.csv')
    # pd.DataFrame(X_test).to_csv('X_test.csv')
    # pd.DataFrame(Y_train).to_csv('Y_train.csv')
    # pd.DataFrame(Y_test).to_csv('Y_test.csv')
    dataset = ECG_Dataset(X_data, Y_data)
    train_size = int((1-dataset_params["RATIO"]) * len(dataset))
    test_size = int(dataset_params["RATIO"] * len(dataset))
    val_size = len(dataset) - train_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # for (inputs, targets) in dataset:
    #     print(f"输入的sigma-delta编码矩阵为\n{inputs}\n对应的类别为第{targets}类")
    train_dataloader = DataLoader(train_dataset, batch_size=training_params["Batch_Size"], shuffle=True, num_workers=16)
    val_dataloader = DataLoader(val_dataset, batch_size=training_params["Batch_Size"], num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=training_params["Batch_Size"], num_workers=16)
    model_train(train_dataloader, test_dataloader, SynNet(n_classes=dataset_params["CLASSES"],n_channels=dataset_params["Time_Partitions"]))
    # model_train(train_dataloader, test_dataloader, SynNet(n_classes=dataset_params["CLASSES"],n_channels=300))
    # model_train(train_dataloader, test_dataloader, WaveSenseNet(dilations=[2, 32],n_classes=dataset_params["CLASSES"],n_channels_in=300))
    # model_train(train_dataloader, test_dataloader,My_net)

if __name__ == '__main__':
    main()