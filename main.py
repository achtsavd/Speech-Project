from cmath import inf
from curses import window
import scipy
import librosa
import numpy as np
import matplotlib.pyplot as plt
import statistics
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error as mse


from scipy.io import wavfile
data, sr = librosa.load("voice.wav")



frame_len, hop_len = 400, 100
frames = librosa.util.frame(data, frame_length=frame_len, hop_length=hop_len)


frames = frames.transpose()
windowed_frames = np.zeros(np.shape(frames))

for i in range(len(frames)):
  windowed_frames[i] = frames[i]*np.hamming(frame_len)




zero_crossings = librosa.zero_crossings(frames,pad=False,zero_pos=False)



nframes ,_ = np.shape(frames)


e = np.zeros((nframes,frame_len))
n = np.zeros((nframes))

po = (frames * np.hamming(frame_len))

po = po**2
po = np.sum(po, axis = 1)

po = np.log(po)
po = po - (np.max(po))

for i in range(len(n)):
    n[i] = np.count_nonzero(zero_crossings[i])



xpoints = np.linspace(1, nframes, nframes)


#for i in range(len(ste[0])):
    #n[i] = np.count_nonzero(zero_crossings[i])

es = po[0:10]
zcs = n[0:10]
esig = np.std(es)
eavg = np.mean(es)
zcsig = np.std(zcs)
zcavg = np.mean(zcs)
izct = zcavg + 3*zcsig
itu = eavg + 6*esig



voiced = np.where(po>itu ,2 , 0)
unvoiced = np.where(n>izct, 1 , 0)
vus = voiced + unvoiced

condlist = [vus==3,vus==2, vus==1, vus==0]
choicelist = [2, 2, 1, 0]
vus = np.select(condlist, choicelist)


def pitch_detection(data):
  auto = sm.tsa.acf(data ,nlags = frame_len)
  peaks = scipy.signal.find_peaks(auto)[0] # Find peaks of the autocorrelation
  lag = peaks[0] # Choose the first peak as our pitch component lag
  pitch = sr / lag
  return pitch


vpitch = np.zeros(np.shape(voiced))
for i in range(len(voiced)):
  if voiced[i] == 0:
    vpitch[i] = -100
  else:
    vpitch[i] = pitch_detection(windowed_frames[i])





figure, axis = plt.subplots(1, 2)


ethreshold = itu*np.ones(len(xpoints))
zthreshold = izct*np.ones(len(xpoints))

print(ethreshold)
axis[0].plot(xpoints, po)
axis[0].plot(xpoints, ethreshold, linestyle='--')
axis[0].set_title("ste")
  
axis[1].plot(xpoints, n)
axis[1].plot(xpoints, zthreshold, linestyle='--')
axis[1].set_title("zcr")

plt.savefig('stezcrnew.png')
plt.close(figure)

figure, axis = plt.subplots(1, 2)



axis[0].plot(data)
axis[0].set_title("wave")

axis[1].plot(xpoints, vus)
axis[1].set_title("voiced/unvoiced/silence")

plt.savefig('vunv.png')
plt.close(figure)

figure, axis = plt.subplots(1, 2)


axis[0].plot(data)
axis[0].set_title("wave")

axis[1].plot(xpoints, vpitch)
axis[1].set_title("pitch")
  

plt.savefig('pitch.png')
plt.close(figure)



vframe = windowed_frames[600]
unframe = windowed_frames[430]
figure, axis = plt.subplots(1, 2)


axis[0].plot(xpoints, po)
axis[0].set_title("ste")
  
axis[1].plot(xpoints, n)
axis[1].set_title("zcr")

plt.savefig('stezcr.png')
plt.close(figure)
def lpc(frame , order):

  a = librosa.lpc(frame, order=order)
  b = np.hstack([[0], -1 * a[1:]])
  y_hat = scipy.signal.lfilter(b, [1], frame)
  


  error = mse(frame, y_hat)
  print("MSE is:", error)

  h = np.array([])
  for x in frame:
    h = np.concatenate((h,np.array([1/(1 - np.sum(np.array([a[_]*x**(-_) for _ in range(len(a))])))])))




  xpoints = np.linspace(1, frame_len, frame_len)

  dft = scipy.fft.fft(frame)

 # plt.plot(xpoints, dft)
  


  figure, axis = plt.subplots(2, 2)


  axis[0][0].plot(frame)
  axis[0][0].plot(y_hat, linestyle='--')
  axis[0][0].set_title("y ,ypred")

  axis[0][1].plot(frame - y_hat)
  axis[0][1].set_title("error")

  axis[1][0].plot(xpoints, dft)
  axis[1][0].set_title("dft")

  axis[1][1].plot(xpoints, h)
  axis[1][1].set_title("H")

  figure.suptitle(order)

  figure.tight_layout()
  plt.show()
  plt.close(figure)
  return error

  plt.show()


verror = ([])
unerror =([])

err = lpc(vframe, 8)
verror.append(err)
err = lpc(vframe, 12)
verror.append(err)
err = lpc(vframe, 16)
verror.append(err)


err = lpc(unframe, 8)
unerror.append(err)
err = lpc(unframe, 12)
unerror.append(err)
err = lpc(unframe, 16)
unerror.append(err)




errpoints = [8,12,16]

figure, axis = plt.subplots(1, 2)

axis[0].plot(errpoints, verror)
axis[0].set_title("voiced")

axis[1].plot(errpoints, unerror)
axis[1].set_title("unvoiced")

plt.show()


