import librosa
import librosa.display
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from scipy import signal

audio_path = "../data/00a7a2f6.wav"
y, sr = librosa.load(audio_path)

print(type(y))
print(y.shape)






X = np.arange(189)

# plt.imshow(D)

# X = librosa.stft(y)
# print(X.shape)


#display waveform
plt.subplot(411)
librosa.display.waveplot(y, sr=sr)


#display spectogram
plt.subplot(412)
Y = librosa.stft(y)
Ydb = librosa.amplitude_to_db(np.abs(Y))
print(Ydb.shape)
librosa.display.specshow(librosa.amplitude_to_db(Ydb, ref=np.max), y_axis='log', x_axis='time')
plt.title('stft')
plt.colorbar(orientation='vertical')

# MFCC - Mel-Frequency Cepstral Coefficients
plt.subplot(413)
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=20)
print(S.shape)
log_S = librosa.amplitude_to_db(S, ref=np.max)
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
plt.title('mel power spectrogram')
plt.colorbar(format='%+02.0f dB')



# Normalization of MFCC

plt.subplot(414)
max_r , min_r = np.max(log_S), np.min(log_S)
print(max_r, min_r)

min_level_db = -100

def normalize(S):
    return np.clip((S-min_level_db)/ -min_level_db, 0, 1)
norm_S = normalize(log_S)
librosa.display.specshow(norm_S, sr=sr, x_axis='time', y_axis='mel')
plt.title('norm mel power spectrogram')
plt.colorbar(format='%+0.1f dB')

plt.tight_layout()
plt.show()

