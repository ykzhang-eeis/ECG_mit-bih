import torch
import numpy as np

try:
    from rich import print
except:
    pass

from torch.utils.data import DataLoader, random_split
from params import dataset_params, training_params
from data_process import loadData
from dataloader import ECG_Dataset
from model import My_net

device = training_params["device"]

model = My_net
state_dict = torch.load("output/model_weights.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

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

correct = 0
total = 0

for batch, target in test_dataloader:
    with torch.no_grad():
        # 确保数据和模型在同一个设备上
        batch = batch.to(torch.float32).to(device)
        target = target.type(torch.LongTensor).to(device)
        out, _, rec = model(batch, record=True)
        peaks = torch.sum(out, dim=1)
        predictions = peaks.argmax(1)
        correct += (predictions == target).sum().item()
        total += target.size(0)

accuracy = correct / total
print(f'Accuracy: {accuracy}')