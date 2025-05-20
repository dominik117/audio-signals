import numpy as np
import simpleaudio as sa

def generate():
    print("🎶 Generating improved audio snippet...")
    fs = 44100
    duration = 2.0
    f1, f2 = 440.0, 660.0

    t = np.linspace(0, duration, int(fs * duration), False)
    tone = 0.5 * (np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t))

    audio = (tone * 32767).astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    play_obj.wait_done()
