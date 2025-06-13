import os
import wave
from typing import Tuple, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from utils.loader import show_stepwise


def load_wav(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Load a WAV file into a numpy array and return (samples, sample_rate).
    Supports 16- and 32-bit PCM.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")
    with wave.open(filepath, 'rb') as wf:
        fs = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        width = wf.getsampwidth()

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width)
    if dtype is None:
        raise ValueError(f"Unsupported sample width: {width}")
    samples = np.frombuffer(frames, dtype=dtype)
    return samples, fs


def plot_waveform(samples: np.ndarray, fs: int, savepath: Optional[str] = None) -> None:
    """
    Plot amplitude vs time for the given samples.
    """
    plt.figure()
    t = np.arange(len(samples)) / fs
    plt.plot(t, samples, linewidth=0.8)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
        print(f"Waveform saved to {savepath}")
    else:
        plt.show()
    plt.close()


def plot_spectrogram(samples: np.ndarray, fs: int,
                     nperseg: int = 1024,
                     noverlap: int = 512,
                     cmap: str = 'viridis',
                     savepath: Optional[str] = None) -> None:
    """
    Plot a log-scale spectrogram of the signal.
    """
    plt.figure()
    f, t, Sxx = signal.spectrogram(samples, fs,
                                   nperseg=nperseg,
                                   noverlap=noverlap,
                                   scaling='density')
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), cmap=cmap)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Spectrogram (dB)')
    plt.colorbar(label='Intensity (dB)')
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)
        print(f"Spectrogram saved to {savepath}")
    else:
        plt.show()
    plt.close()


def compute_time_features(samples: np.ndarray, fs: int,
                          frame_ms: int = 50,
                          hop_ms: int = 25) -> Dict[str, float]:
    """
    Compute global and short-time time-domain features:
      - peak amplitude
      - zero-crossing rate
      - RMS energy
    Returns summary dictionary.
    """
    peak = float(np.max(np.abs(samples)))
    zcr = float(((samples[:-1] * samples[1:]) < 0).mean())
    rms = float(np.sqrt(np.mean(samples.astype(float) ** 2)))
    # short-time energy
    frame_len = int(frame_ms / 1000 * fs)
    hop_len = int(hop_ms / 1000 * fs)
    energies = []
    for start in range(0, len(samples) - frame_len + 1, hop_len):
        frame = samples[start:start + frame_len]
        energies.append(np.sum(frame.astype(float) ** 2))
    avg_energy = float(np.mean(energies)) if energies else 0.0

    return {
        'peak_amplitude': peak,
        'zero_crossing_rate': zcr,
        'global_rms': rms,
        'avg_short_time_energy': avg_energy
    }


def compute_spectral_features(samples: np.ndarray, fs: int,
                              n_fft: int = 2048) -> Dict[str, float]:
    """
    Compute basic spectral features:
      - spectral centroid
      - spectral bandwidth
    """
    S = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1/fs)

    centroid = float((freqs * S).sum() / S.sum())
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * S).sum() / S.sum()))
    return {
        'spectral_centroid': centroid,
        'spectral_bandwidth': bandwidth
    }


def detect_chopin_onsets(samples: np.ndarray, fs: int) -> list:
    """
    Detect Chopin piano note onsets using spectral flux + high-band energy.
    Returns a list of onset times in seconds.
    """
    b, a = signal.butter(4, [200/(fs/2), 0.9], btype='band')
    x = signal.lfilter(b, a, samples)

    win = 1024
    hop = 256
    w = np.hanning(win)
    prev_mag = np.zeros(win//2 + 1)
    freqs = np.fft.rfftfreq(win, 1/fs)

    FLUX_THRESH   = 1e6
    ENERGY_THRESH = 1e5

    raw_onsets = []
    for start in range(0, len(x)-win, hop):
        frame = x[start:start+win] * w
        mag   = np.abs(np.fft.rfft(frame))
        flux  = np.sum(np.clip(mag - prev_mag, 0, None))
        high  = np.sum(mag[freqs > 1000])

        if flux > FLUX_THRESH and high > ENERGY_THRESH:
            time_sec = (start + win//2) / fs
            raw_onsets.append(time_sec)

        prev_mag = mag

    # 4) Merge onsets closer than 50 ms
    onsets = []
    for t in raw_onsets:
        if not onsets or t - onsets[-1] > 0.05:
            onsets.append(t)

    return onsets


def print_pseudo_code():
    pseudocode = [
        "1. LOAD audio signal x[n] at sampling rate fs",
        "",
        "2. PRE-PROCESS",
        "   • Normalize amplitude",
        "   • (Optional) Band-limit to remove sub-200 Hz engine rumble:",
        "         x_lp[n] = bandpass(x, 200 Hz, fs/2)",
        "",
        "3. STFT PARAMETERS",
        "   • window_size = 1024 samples",
        "   • hop_size    = 256 samples",
        "   • prev_mag    = zeros(window_size/2+1)",
        "",
        "4. FOR each frame k = 0, Hop through x_lp with (window_size, hop_size):",
        "     a. frame = x_lp[k : k+window_size] × Hanning",
        "     b. MAG = |FFT(frame)|               # magnitude spectrum",
        "     c. FLUX = sum( max(MAG – prev_mag, 0) )",
        "     d. HIGH_ENERGY = sum( MAG[freq > 1000 Hz] )",
        "     e. IF (FLUX > FLUX_THRESH) AND (HIGH_ENERGY > ENERGY_THRESH):",
        "            mark frame center time as “piano onset”",
        "     f. prev_mag = MAG",
        "",
        "5. AGGREGATE onsets closer than 50 ms into single events",
        "",
        "6. OUTPUT list of piano-note time-stamps",
    ]
    print("\n-- Piano‐Onset Detection Pseudocode --")
    for line in pseudocode:
        print(line)
