import numpy as np
import pyaudio
import wave

from scipy.fft import fftfreq, rfft, irfft
from scipy import signal
import matplotlib.pyplot as plt

# length of data to read (10 seconds)
chunk = 10*44100

# open the file for reading.
wf = wave.open('buonas.wav', 'rb')
fs = wf.getframerate()
p = pyaudio.PyAudio()

# open stream
stream = p.open(format =
                p.get_format_from_width(wf.getsampwidth()),
                channels = wf.getnchannels(),
                rate = fs,
                output = True)

# read data
data = wf.readframes(chunk)
s = np.frombuffer(data, dtype=np.int16) # signal
t = np.linspace(0, chunk/fs, chunk)     # time

# transform to frequency domain
S = rfft(s)
f = fftfreq(s.size, 1/fs)[:s.size//2]

# boundaries
low = np.max(np.argwhere(f.real < 100))
high = np.min(np.argwhere(f.real > 800))

# cut out everything outside boundaries
S[:low] = 0;
S[high:] = 0;

# transform back to time domain
sf = irfft(S)

sf = (sf*(2**15 - 1) / np.max(np.abs(sf))).astype(np.int16)

# play filtered signal
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=fs,
                output=True)

stream.write(sf.tobytes())

# spectrogram
f, t, S = signal.spectrogram(sf, fs, nperseg=2000)
plt.pcolormesh(t, f[0:100], np.log(S[0:100,:]))
plt.ylabel('frequency [Hz]')
plt.xlabel('time [sec]')
plt.show()

# cleanup stuff
wf.close()
stream.close()    
p.terminate()

