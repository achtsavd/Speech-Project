
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


data, sr = librosa.load("voice.wav")



frame_len, hop_len = 160, 80
frames = librosa.util.frame(data, frame_length=frame_len, hop_length=hop_len)
frames = np.transpose(frames)
#frames = (frames * np.hamming(frame_len))


Pxx, freqs, bins, im = plt.specgram(data, window=np.hamming(frame_len), NFFT=frame_len, Fs=sr, noverlap=frame_len-hop_len)


plt.title('10ms window,  5ms step')
plt.savefig('10.png')
