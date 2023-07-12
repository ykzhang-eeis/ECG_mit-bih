import torch
import numpy as np
import wfdb
import pywt
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import json

from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from scipy import signal

# - Import the computational modules and combinators required for the network
from rockpool import TSEvent, TSContinuous
from rockpool.nn.modules import LinearTorch, LIFTorch
from rockpool.parameters import Constant
from rockpool.nn.networks import SynNet,WaveSenseNet
from rockpool.nn.combinators import Sequential, Residual
from rockpool.transform import quantize_methods as q
from pathlib import Path
from params import *
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

# device = "cuda:2" if torch.cuda.is_available() else "cpu"
device = "cpu"

NUM_TAPS = 11 # 数字低通滤波器的阶数，也就是滤波器的长度。
CUT_OFF_FREQ = 15 # 数字低通滤波器的截止频率，单位为赫兹。
    # 函数生成一个数字低通滤波器的系数，其中NUM_TAPS是滤波器的阶数，cutoff是截止频率，fs是采样率。
    # 使用dlti()函数创建一个数字系统，其中第一个参数是滤波器的系数，第二个参数是系统的初始状态，第三个参数是系统的采样时间。
dlti_filter = signal.dlti(signal.firwin(NUM_TAPS, cutoff=CUT_OFF_FREQ, fs=3600.0), [1] + [0] * 10, 1)
    # signal.dimpulse(dlti_filter, n=NUM_TAPS)：使用dimpulse()函数生成数字系统的单位脉冲响应，其中dlti_filter是一个数字系统，n是要生成的单位脉冲响应的长度。
t, imp = signal.dimpulse(dlti_filter, n=NUM_TAPS)

mul = {}
sigma = {}
data = {}
def BSA(input, filter=np.squeeze(imp), threshold=0, channels_num=1):
    
    data = input.copy()
    output = np.zeros(shape=(data.shape[0], data.shape[1]))
    global mul
    global sigma
    global Min
    for i in range(channels_num):
        mul[i]=np.mean(data[i,:])
        sigma[i]=np.std(data[i,:])
        data[i,:]=(data[i,:]-mul[i])/sigma[i]

    for channel in range(1):
        for i in range(data.shape[1]):
            error1 = 0
            error2 = 0
            for j in range(len(filter)):
                if i + j - 1 <= data.shape[1] - 1:
                    error1 += abs(data[channel][i + j - 1] - filter[j])
                    error2 += abs(data[channel][i + j - 1])
            if error1 <= (error2 - threshold):
                output[channel][i] = 1
                for j in range(len(filter)):
                    if i + j - 1 <= data.shape[1] - 1:
                        data[channel][i + j - 1] -= filter[j]
            else:
                output[channel][i] = 0
    output = np.array(output)
    return output

def decoding(spikings, filter=np.squeeze(imp)):
    output = np.zeros(shape=(spikings.shape[0], spikings.shape[1]))
    s = 0
    for channel in range(spikings.shape[0]):
        for t in range(spikings.shape[1]):
            for k in range(len(filter)):
                s += spikings[channel][t - k] * filter[k]
            output[channel][t] = s
            s = 0
    global mul
    global sigma
    global Min
    for channel in range(spikings.shape[0]):
        output[channel,:]=output[channel,:]*(sigma[channel])+mul[channel]
        
    return output


# 测试集在数据集中所占的比例
RATIO = 0.3
# 数据分类类别个数
CLASSES = 4
# ECG信号时间轴分割个数，需要被300整除
Time_Partitions = 15
# ECG信号幅值分割个数
Voltage_Partitions = 16
# Epoch
Num_Epochs = 100
# Batch_size
Batch_Size = 64
# lr
Learning_Rate = 1e-3
# 定义正则化项的系数
lambda_reg = 0.001
# 传入的数据总数
# Num_Datas = 90242 # X_data.shape[0] = 90242
Num_Datas = 640
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
    ecgClassSet = ['N', 'L', 'R', 'V']
    # ecgClassSet = ['L', 'R', 'V']
    

    # 读取心电数据记录
    print("Reading No." + number + " ECG data...")
    """
        wfdb.rdrecord()可以从PhysioNet数据库（或WFDB格式的本地文件）中读取记录数据，并将其返回为一个Record对象
        Record对象包含多个属性，包括信号数据（以Numpy数组的形式存储）和有关记录元数据的信息（例如记录名称、采样率、信号标准化值等）。
    """
    record = wfdb.rdrecord('Dataset/mit-bih-arrhythmia-database-1.0.0/' + number, channel_names=['MLII'])
    data = record.p_signal.flatten() # 将(650000,1)转为(650000, )
    rdata = denoise(data=data)

    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann('Dataset/mit-bih-arrhythmia-database-1.0.0/' + number, 'atr')
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


def loadData():
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    dataSet = []
    lableSet = []
    for n in numberSet:
        getDataSet(n, dataSet, lableSet)

    # 类别均衡前的样本分布情况
    print("Original dataset shape: {}".format(Counter(lableSet)))

    # 过采样
    # ros = RandomOverSampler(random_state=0)
    # X_resampled, y_resampled = ros.fit_resample(dataSet, lableSet)

    # 欠采样
    rus = RandomUnderSampler(random_state=0)
    # normal_samples = 9600
    # abnormal_samples = 4800
    # rus = RandomUnderSampler(sampling_strategy={0:normal_samples, 1:abnormal_samples, 2:abnormal_samples, 3:abnormal_samples}, random_state=0)
    X_resampled, y_resampled = rus.fit_resample(dataSet, lableSet)

    # 类别均衡后的样本分布情况
    print("Resampled dataset shape: {}".format(Counter(y_resampled)))

    # 转numpy数组,打乱顺序
    X_resampled = np.array(X_resampled).reshape(-1, 300)
    y_resampled = np.array(y_resampled).reshape(-1, 1)
    train_ds = np.hstack((X_resampled, y_resampled)) # shape(90242, 301)
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
            # key = torch.tensor(BSA(x_data_row_i.reshape(1,-1))-BSA(-x_data_row_i.reshape(1,-1)))
            # plt.figure()
            # TSEvent.from_raster(torch.transpose(key,0,1), dt=1e-3).plot()
            # plt.show()
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
    # loss = criterion(modSim_peaks, modSim_output)
    # print(confusion_matrix(modSim_peaks, modSim_output))
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
            model.save("output/bsa/model_best.json")

def main():
    X_data, Y_data = loadData()
    print(X_data.shape, Y_data.shape)
    data=X_data
     
    BSA_code1=BSA(data[0,:,:].T,np.squeeze(imp),0) #正值部分
    BSA_code2=BSA(-data[0,:,:].T,np.squeeze(imp),0) #负值部分
    BSA_code=BSA_code1-BSA_code2 
    decode=decoding(BSA_code,np.squeeze(imp)) #解码
    
    plt.figure(0,figsize=(10,3))
    plt.plot(range(1,301),Z_Score_norm(data[0,:,:].T.flatten()), label='y1')
    plt.plot(range(1,len(decode[0,:])+1),Z_Score_norm(decode[0,:]), label='y2')
    plt.show()
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
    # model_train(train_dataloader, test_dataloader, SynNet(n_classes=CLASSES,n_channels=Time_Partitions))
    # # model = SynNet(n_classes=CLASSES,n_channels=300)
    model = SynNet(n_classes=CLASSES,n_channels=dataset_params["Time_Partitions"])
    # # temp = torch.randn((1,1,30)).to(device)
    # # print(model(temp))
    # state_dict = torch.load("output/model_weights.pth")
    # model.load_state_dict(state_dict)
    # g = model.as_graph()
    # spec = x.vA2.mapper(g, weight_dtype='float', threshold_dtype='float', dash_dtype='float')
    # quant_spec = spec.copy()
    # # - Quantize the specification
    # spec.update(q.global_quantize(**spec))
    # config, is_valid, msg = x.vA2.config_from_specification(**spec)
    # print(is_valid)
    # modSim = x.vA2.XyloSim.from_config(config)
    # test_iterator = iter(test_dataloader)
    # batch, target = next(test_iterator)
    # for i in range(batch.shape[0]):
    #     modSim_output, _, r_d = modSim(batch[i].numpy(), record = True)
    #     modSim_peaks = modSim_output.max(1)[0]
    #     print(target[i]==modSim_peaks)
    #导入Samna
    # - Import the Xylo HDK detection function
    # import samna
    # samna.init_samna()
    from rockpool.devices.xylo import find_xylo_hdks
    # from rockpool.devices.xylo import find_xylo_hdks, xylo_devkit_utils
    # xylo_hdk_nodes = xylo_devkit_utils.find_xylo_boards(samna.device_node)
    # - Detect a connected HDK and import the required support package
    connected_hdks, support_modules, chip_versions = find_xylo_hdks()
    found_xylo = len(connected_hdks) > 0
    if found_xylo:
        hdk = connected_hdks[0]
        x = support_modules[0]
    else:
        assert False, 'This tutorial requires a connected Xylo HDK to run.'
    spec = x.mapper(model.as_graph(), weight_dtype = 'float')
    spec.update(q.global_quantize(**spec))
    # - Use rockpool.devices.xylo.config_from_specification
    config, is_valid, msg = x.config_from_specification(**spec)
    # - Use rockpool.devices.xylo.XyloSamna to deploy to the HDK
    if found_xylo:
        modSamna = x.XyloSamna(hdk, config, dt = 0.01)
        print(modSamna)
    # model_train(train_dataloader, test_dataloader, SynNet(n_classes=CLASSES,n_channels=300))

if __name__ == '__main__':
    main()