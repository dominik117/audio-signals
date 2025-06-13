import pyaudio
import wave
import os
import threading
import sys
import numpy as np

def load_wav(filepath) -> (np.ndarray, int):
    """Return (samples, fs)."""
    with wave.open(filepath, 'rb') as wf:
        fs = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    # dtype depends on sampwidth...
    data = np.frombuffer(raw, dtype=np.int16)
    return data, fs

def play_audio(filepath, trim_ms=50):
    if not os.path.exists(filepath):
        print("Audio file not found.")
        return

    print(f"Playing: {filepath} (Press Enter to stop early)")

    try:
        with wave.open(filepath, "rb") as wf:
            framerate = wf.getframerate()
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            nframes = wf.getnframes()

            trim_frames = int((trim_ms / 1000.0) * framerate)
            play_frames = nframes - trim_frames
            if play_frames <= 0:
                print("File too short to trim.")
                return

            wf.rewind()
            frames = wf.readframes(play_frames)

        # Initialize PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(sampwidth),
                        channels=nchannels,
                        rate=framerate,
                        output=True,
                        stream_callback=None)

        # Playback controller
        stop_flag = threading.Event()

        def wait_for_enter():
            input()
            stop_flag.set()

        # Start listening thread
        listener = threading.Thread(target=wait_for_enter, daemon=True)
        listener.start()

        # Chunked playback
        chunk_size = 1024
        index = 0
        while index < len(frames) and not stop_flag.is_set():
            end = index + chunk_size
            stream.write(frames[index:end])
            index = end

        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Playback finished.")

    except Exception as e:
        print(f"Failed to play audio: {e}")
