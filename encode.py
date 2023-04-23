import torch
import numpy as np
from scipy import signal

try:
    from rich import print
except:
    pass

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
def BSA_encoding(input, filter=np.squeeze(imp), threshold=0, channels_num=1):
    
    data = input.copy()
    output = np.zeros(shape=(data.shape[0], data.shape[1]))
    global mul
    global sigma
    global Min
    for i in range(channels_num):
        mul[i]=np.mean(data[i,:])
        sigma[i]=np.std(data[i,:])
        data[i,:]=(data[i,:]-mul[i])/sigma[i]

    for channel in range(channels_num):
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

def BSA_decoding(spikings, filter=np.squeeze(imp)):
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

