import numpy as np
import pyaudio

# adapted from https://stackoverflow.com/questions/8299303/generating-sine-wave-sound-in-python

p = pyaudio.PyAudio()

volume = 0.5    # range [0.0, 1.0]
fs = 44100      # sampling rate, Hz, must be integer
duration = 5.0  # in seconds, may be float
frequency = 440.0       # sine frequency, Hz, may be float

# generate samples, conversion to float32 array
t = np.arange(fs * duration) / fs  # time
samples = (np.sin(2 * np.pi * frequency * t)).astype(np.float32)

# explicitly convert to bytes sequence
output_bytes = (volume * samples).tobytes()

# for paFloat32 sample values must be in range [-1.0, 1.0]
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=fs,
                output=True)

# play
stream.write(output_bytes)
stream.stop_stream()
stream.close()

p.terminate()
