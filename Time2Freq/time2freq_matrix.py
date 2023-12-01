import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy import signal

def read_complex_file(file_path):
    temp = np.genfromtxt(file_path, delimiter=',',dtype='str')
    mapping = np.vectorize(lambda t:complex(t.replace('i','j')))
    data = mapping(temp)
    return data

def time2freq_wvd(time_path, freq_path):
    # 读取时域数据
    signal = read_complex_file(time_path)
    # 转换
    frequency, time, tfr = nk.signal_timefrequency(signal.real,
                                                sampling_rate=500,
                                                method="pwvd")
    plt.close()
    np.savetxt(freq_path, tfr, fmt='%.6e', delimiter=',')

    #################################### PLT #########################################
    ##################################################################################
    # plt.figure(figsize=(3, 3))
    # img = plt.imshow(tfr, aspect='auto')
    # cb = plt.colorbar(img)
    # cb.remove()
    # plt.xticks([])
    # plt.yticks([])
    # plt.axis('off')
    # plt.savefig(freq_path, dpi=1000, bbox_inches='tight', pad_inches=0)
    # plt.close()


def time2freq_stft(time_path, freq_path):
    sig = read_complex_file(time_path)
    nperseg = 50
    f, t, Zxx = signal.stft(sig.real, 500, nperseg=nperseg, noverlap=nperseg//2)
    np.savetxt(freq_path, np.abs(Zxx), fmt='%.6e', delimiter=',')

    #################################### PLT #########################################
    ##################################################################################
    # img = plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    # import ipdb; ipdb.set_trace()
    # cb = plt.colorbar(img)
    # cb.remove()
    # plt.xticks([])
    # plt.yticks([])
    # plt.axis('off')
    # plt.savefig(freq_path, dpi=500, bbox_inches='tight', pad_inches=0)
    # plt.close()

def read_files(src_dir, dest_dir, sub_name):
    root_path = os.path.join(src_dir, sub_name)
    print(root_path)
    for root, dirs, files in os.walk(root_path):
        for file in files:
            print(file)
            file_path = os.path.join(root, file)
            dest_path = os.path.join(dest_dir, sub_name, file)
            time2freq_stft(file_path, dest_path)
            

################################# Time2Freq Dir ##################################
##################################################################################
# src_dir = '/Users/wanghao/Documents/MATLAB/时序数据/series_signal_B500'
# dest_dir = '/Users/wanghao/Documents/MATLAB/频域数据matrix/test'
# read_files(src_dir, dest_dir, 'signal_1')
# read_files(src_dir, dest_dir, 'signal_2')
# read_files(src_dir, dest_dir, 'signal_3')
# read_files(src_dir, dest_dir, 'signal_4')
# read_files(src_dir, dest_dir, 'signal_5')
# read_files(src_dir, dest_dir, 'signal_6')
# read_files(src_dir, dest_dir, 'signal_7')
# read_files(src_dir, dest_dir, 'signal_8')


################################# Time2Freq Single ###############################
##################################################################################
time2freq_stft('./1_sign0_f048.7303_A1.0507.txt', 'stft.txt')
time2freq_wvd('./1_sign0_f048.7303_A1.0507.txt', 'wvd.txt')





