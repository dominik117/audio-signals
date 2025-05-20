import simpleaudio as sa
import wave
import os

def play_audio(filepath, trim_ms=50):
    if not os.path.exists(filepath):
        print("Audio file not found.")
        return

    print(f"Playing: {filepath}")

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

            frames = wf.readframes(play_frames)

        play_obj = sa.play_buffer(frames, nchannels, sampwidth, framerate)
        play_obj.wait_done()
        print("Done.")

    except Exception as e:
        print(f"Failed to play audio: {e}")
