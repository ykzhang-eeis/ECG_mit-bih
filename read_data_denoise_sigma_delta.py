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
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# - Import the computational modules and combinators required for the network
from rockpool import TSEvent, TSContinuous
from rockpool.nn.modules import LinearTorch, LIFTorch
from rockpool.parameters import Constant
from rockpool.nn.networks import SynNet,WaveSenseNet
from rockpool.nn.combinators import Sequential, Residual
from rockpool.transform import quantize_methods as q
from pathlib import Path
from rockpool.devices import xylo as x

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

device = "cuda:2" if torch.cuda.is_available() else "cpu"
# 测试集在数据集中所占的比例
RATIO = 0.3
# 数据分类类别个数
CLASSES = 3
# ECG信号时间轴分割个数，需要被300整除
Time_Partitions = 30
# ECG信号幅值分割个数
Voltage_Partitions = 16
# Epoch
Num_Epochs = 200
# Batch_size
Batch_Size = 1280
# lr
Learning_Rate = 1e-3
# 定义正则化项的系数
lambda_reg = 0.001
# 传入的数据总数
# Num_Datas = 90242 # X_data.shape[0] = 90242
Num_Datas = 256

# 小波去噪预处理
def denoise(data):
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    
    # 得到各层分解系数cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1。其中，cA9是第9层近似系数，cD1~cD9是第1~9层细节系数。
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))

    # 清空两个列表
    cD1.fill(0)
    cD2.fill(0)
    # 对第2~8层的分解系数进行软阈值处理，得到去噪后的系数。
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, X_data, Y_data):
    # ecgClassSet = ['N', 'A', 'V', 'L', 'R']
    # 正常心电(N)、左束支阻滞(L)、右束支阻滞(R)及室性早搏(V)四种心拍类型
    # ecgClassSet = ['N', 'L', 'R', 'V']
    ecgClassSet = ['L', 'R', 'V']
    

    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    """
        wfdb.rdrecord()可以从PhysioNet数据库（或WFDB格式的本地文件）中读取记录数据，并将其返回为一个Record对象
        Record对象包含多个属性，包括信号数据（以Numpy数组的形式存储）和有关记录元数据的信息（例如记录名称、采样率、信号标准化值等）。
    """
    record = wfdb.rdrecord('Dataset/mit-bih-arrhythmia-database-1.0.0/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten() # 将(650000,1)转为(650000, )
    rdata = denoise(data=data)

    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann('mit-bih-arrhythmia-database-1.0.0/' + number, 'atr')
    Rlocation = annotation.sample # shape为(2274,)
    Rclass = annotation.symbol # 长度为2274

    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NAVLR按顺序转换为01234
    while i < j:
        try:
            lable = ecgClassSet.index(Rclass[i])
            x_train = rdata[Rlocation[i] - 99:Rlocation[i] + 201]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return


# 加载数据集并进行预处理
def loadData():
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    dataSet = []
    lableSet = []
    for n in numberSet:
        getDataSet(n, dataSet, lableSet)

    # 转numpy数组,打乱顺序
    dataSet = np.array(dataSet).reshape(-1, 300) # shape(90242, 300)
    lableSet = np.array(lableSet).reshape(-1, 1) # shape(90242, )->(90242, 1)
    train_ds = np.hstack((dataSet, lableSet)) # shape(90242, 301)
    np.random.shuffle(train_ds)

    # 数据集及其标签集
    X = train_ds[:, :300].reshape(-1, 300, 1) # shape(90242, 300, 1)
    Y = train_ds[:, 300] # shape(90242, )

    return X,Y

def sigma_delta_encoding(data, interval_size, num_intervals):

    len_col = int(len(data)/ interval_size)
    # 将原始数据分成大小为(10,30)的矩阵，共10行，每行30列
    data = data.reshape(interval_size, -1)

    # 计算出每个矩阵对应的阈值，比如num_intervals，就按照最大值和最小值等间隔将数值分割为num_intervals份
    # thresholds = torch.linspace(data.min(), data.max(), num_intervals+1)[1:-1] # shape (num_intervals-1, )
    # 如果不在(min,max)做等间隔分得阈值，而是固定范围区间为(-2,6)
    thresholds = torch.linspace(-2, 6, num_intervals+1)[1:-1]


    # print(thresholds)
    data = torch.tensor(data)

    upper_thresh = []
    lower_thresh = []
    for seg_data in data:
        up_counts = 0
        down_counts = 0
        for i in range(len_col-1): # 这个30-1之后要改成采样点除以分割区间个数-1
            for j in range(num_intervals-1):
                if(seg_data[i]<thresholds[j] and seg_data[i+1]>thresholds[j]):
                    up_counts += 1
                if(seg_data[i]>thresholds[j] and seg_data[i+1]<thresholds[j]):
                    down_counts += 1
        upper_thresh.append(up_counts)
        lower_thresh.append(down_counts)

    # 将向上和向下超过阈值的次数填入输出矩阵中
    upper_thresh = torch.tensor(upper_thresh)
    lower_thresh = torch.tensor(lower_thresh)
    output_matrix = torch.stack([upper_thresh, lower_thresh], dim=0)

    return output_matrix

def Z_Score_norm(data):
    eps = 1e-6
    mean = np.mean(data)
    std = np.std(data)
    data_norm = (data - mean) / (std + eps)
    return data_norm

class ECG_Dataset(Dataset):
    def __init__(self, X_data, Y_data):
        super(ECG_Dataset, self).__init__()
        self.data = {}
        for i in range(Num_Datas):
            x_data_row_i = np.array(X_data)[i,:]
            x_data_row_i_norm = Z_Score_norm(x_data_row_i)
            key = sigma_delta_encoding(x_data_row_i_norm, Time_Partitions, Voltage_Partitions)
            plt.figure()
            TSEvent.from_raster(torch.transpose(key,0,1), dt=1e-3).plot()
            plt.show()
            # 不用标准化
            # key = sigma_delta_encoding(x_data_row_i, Time_Partitions, Voltage_Partitions)
            value = Y_data[i]
            self.data[key] = value
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        key = list(self.data.keys())[index]
        value = self.data[key]
        return key, torch.tensor(value)
    
def model_train(train_dataloader, test_dataloader, model=SynNet(4,4)):
    model.to(device)
    criterion = CrossEntropyLoss()
    opt = Adam(model.parameters().astorch(), lr=Learning_Rate)
    best_val_f1 = 0
    for epoch in range(Num_Epochs):
        train_preds = []
        train_targets = []
        sum_loss = 0.0
        for batch, target in tqdm.tqdm(train_dataloader):
            batch = batch.to(torch.float32).to(device)
            target = target.type(torch.LongTensor).to(device)
            model.reset_state()
            opt.zero_grad()
            out, _,rec = model(batch, record = True)
            peaks = out.max(1)[0].to(device)
            loss = criterion(peaks, target)
            l2_reg = 0.0
            for param in model.parameters():
                if(type(param) == str):
                    continue
                else:
                    l2_reg += torch.norm(param)**2
            loss += lambda_reg * l2_reg
            loss.backward()
            opt.step()

            with torch.no_grad():
                pred = peaks.argmax(1).detach()
                train_preds += pred.detach().cpu().numpy().tolist()
                train_targets += target.detach().cpu().numpy().tolist()
                sum_loss += loss.item()/Num_Datas
        sum_f1 = f1_score(train_targets, train_preds, average="macro") # 输出的是所有分类的f1-score
        print(
            f"Train Epoch = {epoch+1}, Loss = {sum_loss}, sum F1 Score = {sum_f1}"
        )
        test_preds = []
        test_targets = []
        test_loss = 0.0
        for batch, target in tqdm.tqdm(test_dataloader):
            with torch.no_grad():
                batch = batch.to(torch.float32).to(device)
                target = target.type(torch.LongTensor).to(device)
                model.reset_state()
                out, _,rec = model(batch, record = True)
                peaks = out.max(1)[0]
                pred = peaks.argmax(1).detach().to(device)
                test_preds += pred.detach().cpu().numpy().tolist()
                test_targets += target.detach().cpu().numpy().tolist()
        f1 = f1_score(test_targets, test_preds, average="macro")
        print(confusion_matrix(test_targets, test_preds))
        test_p, test_r,_,_ =  precision_recall_fscore_support(
            test_targets, test_preds, labels=np.arange(CLASSES)
        )
        print(f"Val Precision = {test_p}, Recall = {test_r}")
        print(f"Val Epoch = {epoch+1}, bestf1score = {best_val_f1}, f1score = {f1}")
        if f1 > best_val_f1:
            best_val_f1 = f1
            model.save("models/model_best.json")

def main():
    # X_train,Y_train为所有的数据集和标签集
    # X_test,Y_test为拆分的测试集和标签集
    X_data, Y_data = loadData()
    print(X_data.shape, Y_data.shape)
    print("data ok!!!")
    X_data = np.reshape(X_data, (-1, 300)) # 将shape从(63170,300,1)转变为(63170,300)
    print(X_data.shape, Y_data.shape)
    # pd.DataFrame(X_train).to_csv('X_train.csv')
    # pd.DataFrame(X_test).to_csv('X_test.csv')
    # pd.DataFrame(Y_train).to_csv('Y_train.csv')
    # pd.DataFrame(Y_test).to_csv('Y_test.csv')

    dataset = ECG_Dataset(X_data, Y_data)
    train_size = int((1-RATIO) * len(dataset))
    test_size = int(RATIO * len(dataset))
    val_size = len(dataset) - train_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    # for (inputs, targets) in dataset:
    #     print(f"输入的sigma-delta编码矩阵为\n{inputs}\n对应的类别为第{targets}类")
    train_dataloader = DataLoader(train_dataset, batch_size=Batch_Size, shuffle=True, num_workers=16)
    val_dataloader = DataLoader(val_dataset, batch_size=Batch_Size, num_workers=16)
    test_dataloader = DataLoader(test_dataset, batch_size=Batch_Size, num_workers=16)
    model_train(train_dataloader, test_dataloader, SynNet(n_classes=CLASSES,n_channels=Time_Partitions))

if __name__ == '__main__':
    main()