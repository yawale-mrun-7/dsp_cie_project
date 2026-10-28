"""
dsp.py: all signal processing for the IIR low pass voice denoiser.
Nothing in here knows about HTTP, so it can also be run from a plain script or notebook.
"""
import base64
import io
import math

import numpy as np
from scipy import signal
from scipy.io import wavfile

MAX_SECONDS = 20  # clips longer than this are truncated to keep the demo snappy
TARGET_FS = 16000  # recordings are resampled to this rate so every clip has a 0-8 kHz spectrum


# ---------------------------------------------------------------- audio I/O
def read_wav(data: bytes):
    """Read WAV bytes -> (sample_rate, mono float array in [-1, 1])."""
    fs, x = wavfile.read(io.BytesIO(data))

    if x.dtype == np.uint8:
        x = (x.astype(np.float64) - 128.0) / 128.0
    elif x.dtype == np.int16:
        x = x / 32768.0
    elif x.dtype == np.int32:
        x = x / 2147483648.0
    else:
        x = x.astype(np.float64)

    if x.ndim > 1:  # stereo -> mono
        x = x.mean(axis=1)

    x = x[: int(MAX_SECONDS * fs)]

    peak = np.max(np.abs(x)) if len(x) else 0.0
    if peak > 0:
        x = x / peak * 0.9  # consistent level, leaves headroom
    return int(fs), x


def resample_to_16k(x: np.ndarray, fs: int) -> np.ndarray:
    """
    Resample a mono float array from fs to TARGET_FS (16000 Hz) using
    scipy.signal.resample_poly, then renormalise peak to 0.9.
    When fs is already 16000 the signal is returned unchanged (no copy).
    """
    if fs == TARGET_FS:
        return x
    # resample_poly needs integer up/down factors; use the GCD to keep them small
    g = math.gcd(TARGET_FS, fs)
    up = TARGET_FS // g
    down = fs // g
    y = signal.resample_poly(x, up, down)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak * 0.9
    return y


def to_wav_base64(x, fs):
    """Float array -> base64 string of a 16-bit WAV (frontend uses it as a data URI)."""
    peak = np.max(np.abs(x))
    if peak > 1.0:  # only scale down if it would clip, so before/after levels stay comparable
        x = x / peak
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, fs, pcm)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------- noise
def add_hiss(clean, fs, snr_db=5.0, noise_start_hz=3500.0, seed=42):
    """
    Add high frequency hiss: white noise passed through a high pass filter,
    scaled so the result has the requested SNR (in dB) against the clean signal.
    A fixed seed keeps the demo repeatable.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(clean))

    start = min(noise_start_hz, 0.9 * fs / 2)
    hp = signal.butter(4, start, btype="high", fs=fs, output="sos")
    noise = signal.sosfilt(hp, noise)

    p_clean = np.mean(clean**2)
    p_noise = np.mean(noise**2)
    scale = np.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10.0)))
    return clean + scale * noise


# ---------------------------------------------------------------- filter design
FILTER_TYPES = ("butterworth", "chebyshev", "elliptic")


def design_sos(ftype, order, cutoff_hz, fs, ripple_db=1.0, stopband_db=40.0):
    """
    Design a low pass IIR filter and return second-order sections (SOS).
    SOS form is used instead of (b, a) because high order direct-form filters
    become numerically unstable; cascaded biquads stay well behaved.
    """
    if not 0 < cutoff_hz < fs / 2:
        raise ValueError(f"Cutoff must be between 0 and {fs / 2:.0f} Hz (Nyquist) for this clip.")
    if not 1 <= order <= 12:
        raise ValueError("Order must be between 1 and 12.")

    if ftype == "butterworth":
        return signal.butter(order, cutoff_hz, btype="low", fs=fs, output="sos")
    if ftype == "chebyshev":
        return signal.cheby1(order, ripple_db, cutoff_hz, btype="low", fs=fs, output="sos")
    if ftype == "elliptic":
        return signal.ellip(order, ripple_db, stopband_db, cutoff_hz, btype="low", fs=fs, output="sos")
    raise ValueError(f"Unknown filter type: {ftype}")


def apply_filter(sos, x, zero_phase=False):
    """Causal filtering by default (what a real-time IIR does). zero_phase runs it forward and backward."""
    if zero_phase:
        return signal.sosfiltfilt(sos, x)
    return signal.sosfilt(sos, x)


# ---------------------------------------------------------------- plot data
def frequency_response(sos, fs, n=512):
    """Magnitude response in dB: (freqs Hz, dB)."""
    w, h = signal.sosfreqz(sos, worN=n, fs=fs)
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-6))
    return w.tolist(), np.maximum(mag_db, -120.0).tolist()


def pole_zero(sos):
    """Poles and zeros of the whole cascade, for an optional pole-zero plot."""
    z, p, _ = signal.sos2zpk(sos)
    return (
        {"re": np.real(z).tolist(), "im": np.imag(z).tolist()},
        {"re": np.real(p).tolist(), "im": np.imag(p).tolist()},
    )


def spectrum(x, fs, n_points=512):
    """Welch power spectral density in dB: (freqs Hz, dB), downsampled for plotting."""
    f, p = signal.welch(x, fs=fs, nperseg=min(2048, len(x)))
    db = 10 * np.log10(np.maximum(p, 1e-14))
    if len(f) > n_points:
        idx = np.linspace(0, len(f) - 1, n_points).astype(int)
        f, db = f[idx], db[idx]
    return f.tolist(), db.tolist()


def waveform(x, fs, n_points=1500):
    """Downsampled waveform for plotting: (time s, amplitude)."""
    step = max(1, len(x) // n_points)
    t = np.arange(0, len(x), step) / fs
    return t.tolist(), x[::step].tolist()


# ---------------------------------------------------------------- metrics
def snr_db(clean, test):
    """SNR of `test` against the clean reference, in dB. None if there is no error at all."""
    err = np.mean((test - clean) ** 2)
    if err == 0:
        return None
    return float(10 * np.log10(np.mean(clean**2) / err))
