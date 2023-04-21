import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


mul = {}
sigma = {}
data = {}

def BSA(input, filter, threshold, channels_num=23):
    """
    :param input: 形状为 [23,1024]
    :param filter: 滤波器
    :param threshold: 阈值
    :return:
    """
    data = input.copy()
    output = np.zeros(shape=(data.shape[0], data.shape[1]))
    global mul
    global sigma
    global Min
    for i in range(channels_num):
        mul[i]=np.mean(data[i,:])
        sigma[i]=np.std(data[i,:])
        data[i,:]=(data[i,:]-mul[i])/sigma[i]
    # for i in range(channels_num):
    #     Min[i] = min(data[i, :])
    #     data[i, :] = data[i, :] - Min[i]
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
    print("BSA编码结束：形状为：",output.shape)
    return output

def decoding(spikings, filter): # (23,1024)
    output = np.zeros(shape=(spikings.shape[0], spikings.shape[1]))
    s = 0
    for channel in range(23): #23
        for t in range(spikings.shape[1]): #1024
            for k in range(len(filter)): #11
                s += spikings[channel][t - k] * filter[k]
            output[channel][t] = s
            s = 0
    global mul
    global sigma
    global Min
    for channel in range(spikings.shape[0]):
        output[channel,:]=output[channel,:]*(sigma[channel])+mul[channel]
    # for channel in range(spikings.shape[0]):
    #     output[channel, :] = output[channel, :] + Min[channel]
    return output

NUM_TAPS = 11
CUT_OFF_FREQ = 15
dlti_filter = signal.dlti(signal.firwin(NUM_TAPS, cutoff=CUT_OFF_FREQ, fs=150.0), [1] + [0] * 10, 1)
t, imp = signal.dimpulse(dlti_filter, n=NUM_TAPS)
 
data=np.random.randn(3,23,1024)
 
BSA_code1=BSA(data[0,:,:],np.squeeze(imp),0) #正值部分
BSA_code2=BSA(-data[0,:,:],np.squeeze(imp),0) #负值部分
BSA_code=BSA_code1-BSA_code2 
decode=decoding(BSA_code,np.squeeze(imp)) #解码
 
 
 
plt.figure(0,figsize=(10,3))
plt.plot(range(1,len(data[0,0,:])+1),data[0,0,:])
plt.plot(range(1,len(decode[0,:])+1),decode[0,:])
plt.show()