import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# - Numpy
import torch
# - Pretty printing
try:
    from rich import print
except:
    pass

# - Disable warnings
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data import DataLoader, random_split
from params import *
from data_process import *
from dataloader import *
from model.my_model import My_net
from rockpool.nn.networks import SynNet

model = SynNet(n_classes=dataset_params["CLASSES"], n_channels=1)
# model = My_net
state_dict = torch.load("output/model_weights1.pth",map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

io_power_list = []
logic_power_list = []
accuracy_list = []

X_data, Y_data = loadData()
X_data = np.reshape(X_data, (-1, 300)) # 将shape从(63170,300,1)转变为(63170,300)
dataset = ECG_Dataset(X_data, Y_data)
train_size = int((1-dataset_params["RATIO"]) * len(dataset))
test_size = int(dataset_params["RATIO"] * len(dataset))
val_size = len(dataset) - train_size - test_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=training_params["Batch_Size"], shuffle=True, num_workers=16)
val_dataloader = DataLoader(val_dataset, batch_size=training_params["Batch_Size"], num_workers=16)
test_dataloader = DataLoader(test_dataset, batch_size=training_params["Batch_Size"], num_workers=16)
test_iterator = iter(test_dataloader)
batch, target = next(test_iterator)
batch = batch.to(torch.float32)
target = target.type(torch.LongTensor)
out, _,rec = model(batch, record = True)
peaks = torch.sum(out,dim=1)
print(peaks.argmax(1)==target)
# for i in range(batch.shape[0]):
#     output = model(batch[i].float())[0]
#     peaks = torch.sum(output, dim=1).argmax()
#     print(peaks,target[i])
#     accuracy_list.append(target[i].long()==peaks)
# print(accuracy_list)