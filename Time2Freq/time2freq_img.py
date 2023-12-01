import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import neurokit2 as nk
rng = np.random.default_rng()

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
    plt.figure(figsize=(4, 3))
    img = plt.imshow(tfr, aspect='auto')
    cb = plt.colorbar(img)
    cb.remove()
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    # plt.show()
    plt.savefig(freq_path, dpi=1000, bbox_inches='tight', pad_inches=0)
    plt.close()


def time2freq_stft(time_path, freq_path):
    sig = read_complex_file(time_path)
    nperseg = 50
    f, t, Zxx = signal.stft(sig.real, 500, nperseg=nperseg, noverlap=nperseg//2)
    img = plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    cb = plt.colorbar(img)
    cb.remove()
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    # plt.show()
    plt.savefig(freq_path, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()


time2freq_wvd('./1_sign0_f048.7303_A1.0507.txt', './wvd.png')
time2freq_stft('./1_sign0_f048.7303_A1.0507.txt', './stft.png')



