import numpy as np
import pyaudio

volume = 0.4

def play_audio(samples, fs=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)
    stream.write(samples.tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

def play_tone():
    freq = float(input("Enter frequency in Hz (e.g. 440): "))
    duration = float(input("Enter duration in seconds (e.g. 2): "))
    fs = 44100
    t = np.arange(int(fs * duration)) / fs
    wave = (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    play_audio(wave)

def play_chord():
    freqs = input("Enter frequencies (space-separated, e.g. 440 550 660): ")
    freqs = [float(f) for f in freqs.strip().split()]
    duration = float(input("Enter duration in seconds (e.g. 2): "))
    fs = 44100
    t = np.arange(int(fs * duration)) / fs
    wave = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    wave = (volume * wave / len(freqs)).astype(np.float32)
    play_audio(wave)

def play_sweep():
    start_freq = float(input("Start frequency (e.g. 300): "))
    end_freq = float(input("End frequency (e.g. 800): "))
    duration = float(input("Duration in seconds (e.g. 3): "))
    fs = 44100
    t = np.linspace(0, duration, int(fs * duration), False)
    sweep = np.sin(2 * np.pi * ((start_freq + (end_freq - start_freq) * t / duration) * t))
    wave = (volume * sweep).astype(np.float32)
    play_audio(wave)

def run_playground():
    while True:
        print("\n-- Sine Wave Playground --")
        print("1. Play a pure tone")
        print("2. Create a chord")
        print("3. Generate a sweep")
        print("4. Back")
        choice = input("> ")
        if choice == "1":
            play_tone()
        elif choice == "2":
            play_chord()
        elif choice == "3":
            play_sweep()
        elif choice == "4":
            break
        else:
            print("Invalid input.")
