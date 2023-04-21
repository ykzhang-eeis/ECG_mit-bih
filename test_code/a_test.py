"""
    MIT-BIH数据集是以WFDB格式保存的，包括.hea, .dat, .atr文件等
    示例代码是通过wfdb包来读取数据
"""
import numpy as np
import wfdb
import pywt # PyWavelet库，是用于处理小波变换的Python库，提供了多种小波基函数
import pandas as pd
import matplotlib.pyplot as plt
import torch

# 测试集在数据集中所占的比例
RATIO = 0.3

# 小波去噪预处理
def denoise(data):
    """
        pywt.wavedec()是Python信号处理库PyWavelets的一个函数，用于对给定的信号进行小波分解。
        小波分解是一种将信号分解为一组小波基函数的线性组合的过程
        pywt.wavedec()函数接受三个必要参数：data、wavelet和level。其中:
        data是要进行小波分解的信号序列；
        wavelet是小波基函数的名称或一个小波对象；
        level是小波分解的级数。
        函数返回一个包含小波分解系数的元组，其中第一个元素是逼近系数数组，其余元素是细节系数数组，
        数量等于小波分解的级数。这些系数数组可以用于重构原始信号。
    """
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    # 得到各层分解系数cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1。其中，cA9是第9层近似系数，cD1~cD9是第1~9层细节系数。
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    """
        np.median(np.abs(cD1))计算cD1的绝对值中位数，0.6745是标准正态分布的0.75分位数，
        np.sqrt(2 * np.log(len(cD1)))是根据Donoho提出的经验规则计算得出的标准差估计值，
        最后通过两个值的乘积计算出阈值。这是一种基于数据特性的软阈值方法，用于小波去噪中。
    """
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
    print("正在读取 " + number + " 号心电数据...")
    """
        wfdb.rdrecord()可以从PhysioNet数据库（或WFDB格式的本地文件）中读取记录数据，并将其返回为一个Record对象
        Record对象包含多个属性，包括信号数据（以Numpy数组的形式存储）和有关记录元数据的信息（例如记录名称、采样率、信号标准化值等）。
    """
    record = wfdb.rdrecord('mit-bih-arrhythmia-database-1.0.0/' + number, channel_names=['MLII'])
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

    # 测试集及其标签集
    shuffle_index = np.random.permutation(len(X)) # shape(90242, )
    test_length = int(RATIO * len(shuffle_index)) # 90242 * 0.3 = 27072
    test_index = shuffle_index[:test_length] # shape(27072, )
    train_index = shuffle_index[test_length:] # shape(61370, )
    X_test, Y_test = X[test_index], Y[test_index] # shape(27072, 300, 1), shape(27072, )
    X_train, Y_train = X[train_index], Y[train_index] # shape(61370, 300, 1), shape(61370, )

    return X_train, Y_train, X_test, Y_test

def sigma_delta_encoding(data, interval_size, num_intervals):

    len_col = int(len(data)/ interval_size)
    # 将原始数据分成大小为(10,30)的矩阵，共10行，每行30列
    data = data.reshape(interval_size, -1)

    # 计算出每个矩阵对应的阈值，比如num_intervals，就按照最大值和最小值等间隔将数值分割为num_intervals份
    thresholds = torch.linspace(data.min(), data.max(), num_intervals+1)[1:-1] # shape (num_intervals-1, )
    
    # 对于每个矩阵，将其数据点映射到相应的阈值中，并将映射后的数据点转换成比特码，得到每个矩阵对应的sigma-delta编码序列
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


def main():
    # X_train,Y_train为所有的数据集和标签集
    # X_test,Y_test为拆分的测试集和标签集
    X_train, Y_train, X_test, Y_test = loadData()
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    print("data ok!!!")
    X_train = np.reshape(X_train, (-1, 300)) # 将shape从(63170,300,1)转变为(63170,300)
    X_test = np.reshape(X_test, (-1,300))
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    # pd.DataFrame(X_train).to_csv('X_train.csv')
    # pd.DataFrame(X_test).to_csv('X_test.csv')
    # pd.DataFrame(Y_train).to_csv('Y_train.csv')
    # pd.DataFrame(Y_test).to_csv('Y_test.csv')
    data_x_train_0 = np.array(X_train)[0,:]
    eps = 1e-6
    mean0 = np.mean(data_x_train_0)
    std0 = np.std(data_x_train_0)
    data_x_train_0_norm = (data_x_train_0 - mean0) / (std0 + eps)
    res = sigma_delta_encoding(data_x_train_0_norm, 15, 8)
    print(res)
    

if __name__ == '__main__':
    main()