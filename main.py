import numpy as np

from torch.utils.data import DataLoader, random_split
from data_process import loadData
from params import dataset_params, training_params
from dataloader import ECG_Dataset
from train import model_train_snn
from model import My_net

try:
    from rich import print
except:
    pass

def main():
    X_data, Y_data = loadData()
    X_data = np.reshape(X_data, (-1, 300)) # 将shape从(len, 300, 1)转变为(len, 300)
    print(X_data.shape, Y_data.shape)
    print("data ok!!!")
    dataset = ECG_Dataset(X_data, Y_data) # 这部分要运行很久
    train_size = int((1-dataset_params["RATIO"]) * len(dataset))
    test_size = int(dataset_params["RATIO"] * len(dataset))
    val_size = len(dataset) - train_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=training_params["Batch_Size"], shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=training_params["Batch_Size"], num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=training_params["Batch_Size"], shuffle=False, num_workers=0)
    # model_train_snn(train_dataloader, test_dataloader, SynNet(n_classes=dataset_params["CLASSES"],n_channels=dataset_params["Time_Partitions"]))
    model_train_snn(train_dataloader, test_dataloader, My_net)

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