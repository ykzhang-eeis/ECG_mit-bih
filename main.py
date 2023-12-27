import numpy as np

from torch.utils.data import DataLoader, random_split
from data_process import load_and_preprocess_data
from params import dataset_params, training_params
from dataloader import ECGDataset
from train import train_snn_model
from model import MyNet
from inference import run_inference


def main():
    X_data, Y_data = load_and_preprocess_data()
   
    dataset = ECGDataset(X_data, Y_data, training_params["Num_Datas"], dataset_params["Time_Partitions"], 
                         dataset_params["Voltage_Partitions"])
    
    train_size = int((1 - dataset_params["RATIO"]) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=training_params["Batch_Size"], 
                                    shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = MyNet
    train_snn_model(train_dataloader, test_dataloader, model)
    infer_acc = run_inference(model, test_dataloader)
    # print(f'Inference Accuracy: {infer_acc:.4f}')

    # from deploy2Xylo import xylo_inference
    # xylo_inference(test_dataloader)

    # from test_xylosim import xyloSim_inference
    # xyloSim_inference(test_dataloader)
if __name__ == '__main__':
    main()