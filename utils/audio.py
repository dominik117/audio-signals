import pyaudio
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

            # Rewind to start and read only the trimmed frame range
            wf.rewind()
            frames = wf.readframes(play_frames)

        # Initialize PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(sampwidth),
                        channels=nchannels,
                        rate=framerate,
                        output=True)

        stream.write(frames)
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("Done.")

    except Exception as e:
        print(f"Failed to play audio: {e}")
