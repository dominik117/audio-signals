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

# choose template to locate 
template = s[int(1.2*fs):int(1.8*fs)]

# play sample
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=fs,
                output=True)

# play
stream.write(template.tobytes())

corr = signal.fftconvolve(s, template, mode='same') 

# show correlation
plt.plot( t, corr )
plt.ylabel('correlation')
plt.xlabel('time [sec]')
plt.show()

# cleanup stuff
wf.close()
stream.close()    
p.terminate()

