import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm.auto import tqdm 
from pathlib import Path
from rockpool.devices.xylo.syns61201 import config_from_specification, xa2_devkit_utils as hdu

import torch
from rockpool.nn.modules import LinearTorch, LIFTorch
from rockpool.parameters import Constant
# - Matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams['figure.dpi'] = 300

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
from rockpool.nn.networks.wavesense import WaveSenseNet
from rockpool.transform import quantize_methods as q

#获取模型
dilations = [2, 32]
n_out_neurons = 2
n_inp_neurons = 4
n_neurons = 16
kernel_size = 2
tau_mem = 0.002
base_tau_syn = 0.002
tau_lp = 0.01
threshold = 0.6
dt = 0.001
sim = WaveSenseNet(
    dilations=dilations,
    n_classes=n_out_neurons,
    n_channels_in=n_inp_neurons,#in_channel
    n_channels_res=n_neurons,
    n_channels_skip=n_neurons,
    n_hidden=n_neurons,
    kernel_size=kernel_size,
    bias=Constant(0.0),
    smooth_output=True,
    tau_mem=Constant(tau_mem),
    base_tau_syn=base_tau_syn,
    tau_lp=tau_lp,
    threshold=Constant(threshold),
    neuron_model=LIFTorch,
    dt=dt,
)
# sim.load('/home/ruixing/workspace/chbtar/chb/models/SNN_model_Vmem.pth')

#导入Samna
# - Import the Xylo HDK detection function
from rockpool.devices.xylo import find_xylo_hdks

# - Detect a connected HDK and import the required support package
connected_hdks, support_modules, chip_versions = find_xylo_hdks()

found_xylo = len(connected_hdks) > 0

if found_xylo:
    hdk = connected_hdks[0]
    x = support_modules[0]
else:
    assert False, 'This tutorial requires a connected Xylo HDK to run.'
spec = x.mapper(sim.as_graph(), weight_dtype = 'float')
spec.update(q.global_quantize(**spec))
# - Use rockpool.devices.xylo.config_from_specification
config, is_valid, msg = x.config_from_specification(**spec)
# - Use rockpool.devices.xylo.XyloSamna to deploy to the HDK
if found_xylo:
    modSamna = x.XyloSamna(hdk, config, dt = dt)
    print(modSamna)

# #导入待测mne数据
# sampling_frequency = 250
# data = mne.io.read_raw_edf('raw_selected/raw_selected/train/chb01_03.edf',preload=True)
# #'chb02_16': [2972, 3053]
# data = data.filter(l_freq=None,h_freq=30)
# data = data.resample(sfreq=sampling_frequency)
# tmin = 2985
# tmax = 3045+5
# data = data.crop(tmin=tmin,tmax=tmax)
# picks1= ['C3-P3']
# picks1= ['C3-P3']
# picks1= ['C3-P3']
# picks2= ['C4-P4']
# one_sample_data = []
# spike_array_one_trail = []
# picks_list = list([picks1, picks2])
# time_vector = np.linspace(0,5,1250)
# interpfact = 250 #连续化时的采样率
# refractory = 3e-4
# dt = 0.01
# sample = []
# spike_array = []
# epilepsy_test = []
# io_power_list = []
# logic_power_list = []
# for j in picks_list:
#     data_one_channel = data.get_data(units='uV',picks=j)
#     one_sample_data.append(data_one_channel)
# one_sample_data = np.asarray(one_sample_data)
# array = one_sample_data.reshape(2,-1)
# start =0
# len_windows = 1250
# step = 125
# end = len_windows
# array_encode = array.copy()
# trigger = 0
# n = 0
    
# print(f'Clock freq. set to {hdu.set_xylo_core_clock_freq(modSamna._device, 6.75)} MHz')
# for i in tqdm(range(array.shape[1]//125)):
#     array_slices = array_encode[:,start:end]
#     if array_slices.shape[1] != 1250:
#         break
#     array_slices.reshape(2,1250)
#     for elc in np.arange(2):
#         data_one_channel = array_slices[elc]
#         ecog_signal, ecog_spikes = pre_process_data(raw_signal = data_one_channel,
#                                                 time_vector = time_vector, 
#                                                 interpfact = interpfact, 
#                                                 refractory = refractory)
#         spike_list = {}
#         spike_list['up'] = ecog_spikes['up']
#         spike_list['dn'] = ecog_spikes['dn']
#         spiketimes, neuronID = concatenate_spikes(spike_list)
#         if spiketimes.shape[0] != 0:
#             if spiketimes[-1] == (5.):
#                 spiketimes = np.delete(spiketimes,[-1])
#                 neuronID = np.delete(neuronID,[-1])
                
#         if spiketimes.size == 0 or spiketimes.size == 1:
#             synchronous_input = torch.zeros(1,500,2)
#         else:
#             # Get input signal 
#             dt_original_data = 1/sampling_frequency
#             num_timesteps = int(np.around(ecog_signal['time'][-1]+ dt_original_data - ecog_signal['time'][0], decimals = 3)* (1/dt))
#             t_start  = ecog_signal['time'][0]
#             t_stop = ecog_signal['time'][-1]
#             asynchronous_input, synchronous_input = get_input_spikes(spiketimes = spiketimes,
#                                                                 neuronID = neuronID,
#                                                                 t_start = t_start, 
#                                                                 t_stop = t_stop,
#                                                                 num_timesteps = num_timesteps,
#                                                                 dt = dt)

#         synchronous_input = synchronous_input.squeeze()
#         synchronous_input = np.asarray(synchronous_input)
#         synchronous_input = synchronous_input.T
#         spike_array_one_trail.append(synchronous_input)
#     spike_array_one_trail = np.asarray(spike_array_one_trail)
#     spike_array_one_trail = spike_array_one_trail.reshape(4,500)
#     start += step
#     end += step
#     #送进Samna
#     tensor = torch.from_numpy(spike_array_one_trail.T)
#     data = torch.tensor(tensor,dtype=torch.float)
#     data = torch.reshape(data,(500,4))
#     data = data.numpy()
#     data = data.astype(int)
#     modSamna.reset_state()
#     # np.save('2_16_t1_test.npy',spike_array_one_trail)
#     #************************************#
#     out, _, recordings = modSamna((data*3).clip(0, 15),record=False,record_power = True,read_timeout = 40)
#     #************************************#
#     print(recordings)
#     io_power = np.mean(recordings['io_power'])
#     logic_power = np.mean(recordings['logic_power'])
#     print('io_power:',io_power,'logic_power:',logic_power)
#     io_power_list.append(io_power)
#     logic_power_list.append(logic_power)
#     spike_array_one_trail = []
#     n = n+1
# print('完成')
# print(n,len(io_power_list),len(logic_power_list))



# # 创建一个figure对象和两个子图，figsize参数指定图形大小
# fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
# # 在第一个子图上绘制io_power_list曲线
# x0 = np.linspace(0, len(io_power_list)-1, num=1000, endpoint=True)
# f0 = interp1d(range(len(io_power_list)), io_power_list, kind='cubic')
# y_smooth0 = f0(x0)

# # 绘制平滑曲线
# axs[0].plot(x0, y_smooth0, label='io Power', color='#1E90FF', linewidth=1)


# # 添加横轴标签
# axs[0].set_xlabel('Index')

# # 添加纵轴标签
# axs[0].set_ylabel('Power (W)')

# # 添加标题
# axs[0].set_title('IO Power', fontsize=14)

# # 添加图例
# axs[0].legend(loc='lower left',fontsize=8)

# # 自动调整子图间距
# plt.tight_layout()

# # 在第二个子图上绘制logic_power_list曲线
# x1 = np.linspace(0, len(logic_power_list)-1, num=1000, endpoint=True)
# f1 = interp1d(range(len(logic_power_list)), logic_power_list, kind='cubic')
# y_smooth1 = f1(x1)

# # 绘制平滑曲线
# axs[1].plot(x1, y_smooth1, label='Logic Power', color='darkorange', linewidth=1)


# # 添加横轴标签
# axs[1].set_xlabel('Index')

# # 添加纵轴标签
# axs[1].set_ylabel('Power (W)')

# # 添加标题
# axs[1].set_title('Logic Power', fontsize=14)

# # 添加图例
# axs[1].legend(loc='lower left',fontsize=8)
# # 自动调整子图间距
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.3)

# for ax in axs:
#     ax.grid(alpha=0.4, linestyle='--', linewidth=0.5)
# # 显示图形
# plt.show()


# fig.savefig('power.jpg',dpi = 2000)
# average_logic = sum(logic_power_list) / len(logic_power_list)
# average_io = sum(io_power_list) / len(io_power_list)
# print('io_power:',average_io)
# print('logic_power:',average_logic)