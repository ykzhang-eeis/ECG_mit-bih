import torch
import numpy as np
from torch.utils.data import Dataset
from params import training_params, dataset_params
from data_process import Z_Score_norm, sigma_delta_encoding

class ECG_Dataset(Dataset):
    def __init__(self, X_data, Y_data):
        super(ECG_Dataset, self).__init__()
        self.data = {}
        for i in range(training_params["Num_Datas"]):
            x_data_row_i = np.array(X_data)[i,:]
            x_data_row_i_norm = Z_Score_norm(x_data_row_i)
            # plt.plot(x_data_row_i_norm)
            # plt.show()
            key = sigma_delta_encoding(x_data_row_i_norm, dataset_params["Time_Partitions"], dataset_params["Voltage_Partitions"])
            key = key.reshape(-1, 15)
            value = Y_data[i]
            self.data[key] = value
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        key = list(self.data.keys())[index]
        value = self.data[key]
        return key, torch.tensor(value)