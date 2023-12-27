import torch

dataset_params = {
    "RATIO": 0.3, # 测试集在数据集中所占的比例
    "CLASSES": 4, # 数据分类类别个数
    "Time_Partitions": 60, # ECG信号时间轴分割个数，需要被300整除
    "Voltage_Partitions": 64 # ECG信号幅值分割个数
}

training_params = {
    "Num_Datas": 1280, # 传入的数据总数X_data.shape[0] = 90242(过采样之前，过采样之后为286892)
    "Num_Epochs": 100,
    "Batch_Size": 64,
    "Learning_Rate": 1e-3,
    "lambda_reg": 0.001, # 定义正则化项的系数
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
