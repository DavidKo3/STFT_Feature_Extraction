#-*- coding:utf-8 -*-

import librosa
import librosa.display
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from scipy import signal


data = "../data/musdb18_1.mp4"
y, sr  = librosa.load(data)
time = np.linspace(0, len(y)/sr, len(y)) # time axis

print("length of data : ", len(y))


def down_sampling(input_wav, origin_sr, resample_sr):
    y, sr = librosa.load(input_wav, sr=origin_sr)
    resample = librosa.resample(y, sr, resample_sr)
    print("original wav sr: {}, original wav shape: {}, "
          "resample wav sr: {}, resample shape: {}".format(origin_sr, y.shape, resample_sr, resample.shape))
    return resample

resample = down_sampling(data, origin_sr=sr, resample_sr=sr/2)

#
plt.figure(figsize=(10, 4)) # 10, 4
plt.subplot(2, 1, 1)
time1 = np.linspace(0, len(y)/sr, len(y))
plt.plot(time1, y)

plt.subplot(2, 1, 2)

resample_sr = 8000
print("sr : ", sr)
print("len of resample : ", len(resample))

time2 = np.linspace(0, len(resample)/resample_sr, len(resample))
plt.plot(time2, resample)
plt.title("Resampled Wav")

plt.tight_layout()
plt.savefig('comparare_16k_vs_8k.png')





































