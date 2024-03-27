import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

from .data_process import Z_score_norm, sigma_delta_encoding


class ECGDataset(Dataset):
    def __init__(self, X_data: np.ndarray, Y_data: np.ndarray, 
                 num_datas: int, time_partitions: int, voltage_partitions: int):
        super().__init__()
        self.samples = []
        X_data = np.reshape(X_data, (-1, 300)) # Reshape from (len, 300, 1) to (len, 300)

        for i in range(num_datas):
            x_data_row_i = X_data[i, :]
            x_data_row_i_norm = Z_score_norm(x_data_row_i)
            key = sigma_delta_encoding(x_data_row_i_norm, time_partitions, voltage_partitions)
            value = Y_data[i]
            self.samples.append((key, value))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        key, value = self.samples[index]
        return torch.tensor(key, dtype=torch.float32), torch.tensor(value, dtype=torch.long)
