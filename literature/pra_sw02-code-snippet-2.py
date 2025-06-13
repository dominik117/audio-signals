import numpy as np
import pyaudio
import wave

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

# play the actual chunk
stream.write(data)

# spectrogram
f, t, S = signal.spectrogram(s, fs, nperseg=2000)
plt.pcolormesh(t, f[0:100], np.log(S[0:100,:]),shading='auto')
plt.ylabel('frequency [Hz]')
plt.xlabel('time [sec]')
plt.show()

# cleanup stuff
wf.close()
stream.close()    
p.terminate()
