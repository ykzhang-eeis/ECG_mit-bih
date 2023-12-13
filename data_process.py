import numpy as np
import torch
import wfdb
import pywt
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
try:
    from rich import print
except:
    pass

def sigma_delta_encoding(data, interval_size, num_intervals):

    data = data.reshape(interval_size, -1)

    # 计算出每个矩阵对应的阈值，比如num_intervals，就按照最大值和最小值等间隔将数值分割为num_intervals份
    # thresholds = torch.linspace(data.min(), data.max(), num_intervals+1)[1:-1] # shape (num_intervals-1, )
    # 如果不在(min,max)做等间隔分得阈值，而是固定范围区间为(-2,6)
    thresholds = torch.linspace(-2, 6, num_intervals+1)[1:-1]

    # print(thresholds)
    data = torch.tensor(data)

    # 计算每一列与阈值的比较结果
    # 为了进行向量化比较，需要扩展数据和阈值的维度以便广播
    data_expanded = data.unsqueeze(2)  # shape: (interval_size, len_col, 1)
    thresholds_expanded = thresholds.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, num_intervals-1)

    # 计算是否上升或下降超过阈值
    upper_cross = (data_expanded[:, :-1] < thresholds_expanded) & (data_expanded[:, 1:] > thresholds_expanded)
    lower_cross = (data_expanded[:, :-1] > thresholds_expanded) & (data_expanded[:, 1:] < thresholds_expanded)

    # 计算每个区间的上升和下降次数
    upper_thresh_counts = upper_cross.sum(dim=1).sum(dim=1)
    lower_thresh_counts = lower_cross.sum(dim=1).sum(dim=1)

    # 将结果合并为一个输出矩阵
    output_matrix = torch.stack([upper_thresh_counts, lower_thresh_counts], dim=0)

    return output_matrix

# 小波去噪预处理
def denoise(data):
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    
    # 得到各层分解系数cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1。其中，cA9是第9层近似系数，cD1~cD9是第1~9层细节系数。
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))

    cD1.fill(0)
    cD2.fill(0)
    # 对第2~8层的分解系数进行软阈值处理，得到去噪后的系数。
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

def Z_Score_norm(data):
    eps = 1e-6
    mean = np.mean(data)
    std = np.std(data)
    data_norm = (data - mean) / (std + eps)
    return data_norm

# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, X_data, Y_data):

    # 正常心电(N)、左束支阻滞(L)、右束支阻滞(R)及室性早搏(V)四种心拍类型
    ecgClassSet = ['N', 'L', 'R', 'V']
    
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

    # 因为只选择NVLR四种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NVLR按顺序转换为0123
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

    # 类别均衡前的样本分布情况
    print("Original dataset shape: {}".format(Counter(lableSet)))

    # 欠采样
    rus = RandomUnderSampler(random_state=0)
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
