"""
Writes a synthetic, voice-like WAV (harmonics with a moving pitch and syllable-like
amplitude bursts) into samples/ so you can test the backend before recording real clips.
Replace it with real speech recordings for the actual demo.
"""
from pathlib import Path

import numpy as np
from scipy.io import wavfile

fs = 16000
dur = 6.0
t = np.arange(int(fs * dur)) / fs

f0 = 120 + 25 * np.sin(2 * np.pi * 0.4 * t)          # slowly moving pitch
phase = 2 * np.pi * np.cumsum(f0) / fs
x = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 25))  # harmonics up to ~3 kHz
env = np.clip(np.sin(2 * np.pi * 2.0 * t), 0, None) ** 0.7     # syllable-like bursts
x = x * env
x = 0.8 * x / np.max(np.abs(x))

out = Path(__file__).parent / "samples"
out.mkdir(exist_ok=True)
wavfile.write(out / "synthetic_voice.wav", fs, (x * 32767).astype(np.int16))
print("wrote", out / "synthetic_voice.wav")
