"""
test_backend.py: automated tests for the IIR low-pass voice denoiser backend.

Run from the backend/ directory with:
    .\\venv\\Scripts\\python.exe test_backend.py

Or with pytest if installed:
    .\\venv\\Scripts\\python.exe -m pytest test_backend.py -v
"""
import io
import json
import math
import os
import struct
import sys

import numpy as np
from scipy.io import wavfile
from fastapi.testclient import TestClient

# Make sure imports resolve from the backend directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import app
import dsp

client = TestClient(app)


# ---------- helper: create synthetic WAV bytes ----------
def make_wav_bytes(fs=48000, duration=3.0, channels=1, freq=440.0):
    """Generate a sine-wave WAV at the given sample rate and channel count."""
    n = int(fs * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    x = (0.5 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    if channels == 2:
        x = np.column_stack([x, x])
    buf = io.BytesIO()
    wavfile.write(buf, fs, x)
    return buf.getvalue()


def make_silent_wav(fs=16000, duration=2.0):
    """WAV of nearly pure silence."""
    n = int(fs * duration)
    x = (np.ones(n) * 0.001).astype(np.float64)  # very quiet
    pcm = (x * 32767 * 1e-5).astype(np.int16)     # RMS well below 1e-4
    buf = io.BytesIO()
    wavfile.write(buf, fs, pcm)
    return buf.getvalue()


def make_short_wav(fs=16000, duration=0.3):
    """WAV that is too short (under 1 second)."""
    return make_wav_bytes(fs=fs, duration=duration)


def make_non_wav():
    """Not a WAV file at all."""
    return b"This is not a WAV file at all, just plain text."


# ================================================================ TESTS
class TestHealth:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestSamples:
    def test_list_samples(self):
        r = client.get("/samples")
        assert r.status_code == 200
        assert "samples" in r.json()


class TestGenerateNoisy:
    def test_generate(self):
        samples = client.get("/samples").json()["samples"]
        if not samples:
            return  # no sample files on disk
        r = client.post("/generate-noisy", json={
            "sample": samples[0], "snr_db": 5, "noise_start_hz": 3500
        })
        assert r.status_code == 200
        p = r.json()
        assert "audio_id" in p
        assert p["fs"] > 0
        assert p["has_reference"] is True
        assert p["noisy_audio"]  # non-empty base64


class TestUpload:
    def test_upload_wav(self):
        wav = make_wav_bytes(fs=16000, duration=2.0)
        r = client.post("/upload", files={"file": ("test.wav", wav, "audio/wav")})
        assert r.status_code == 200
        p = r.json()
        assert "audio_id" in p
        assert p["has_reference"] is False  # upload never has a clean reference

    def test_upload_non_wav(self):
        r = client.post("/upload", files={"file": ("test.txt", make_non_wav(), "text/plain")})
        assert r.status_code == 400


class TestRecord:
    def test_record_clean_48k_stereo(self):
        """48 kHz stereo WAV, treat_as=clean: should get fs=16000 and has_reference=True."""
        wav = make_wav_bytes(fs=48000, duration=3.0, channels=2)
        r = client.post("/record", data={
            "treat_as": "clean", "snr_db": "5", "noise_start_hz": "3500"
        }, files={"file": ("rec.wav", wav, "audio/wav")})
        assert r.status_code == 200
        p = r.json()
        assert p["fs"] == 16000
        assert "audio_id" in p
        assert p["has_reference"] is True

    def test_record_noisy(self):
        """treat_as=noisy: has_reference should be False."""
        wav = make_wav_bytes(fs=44100, duration=2.0)
        r = client.post("/record", data={
            "treat_as": "noisy"
        }, files={"file": ("rec.wav", wav, "audio/wav")})
        assert r.status_code == 200
        p = r.json()
        assert p["has_reference"] is False

    def test_record_reject_silent(self):
        wav = make_silent_wav()
        r = client.post("/record", data={"treat_as": "clean"},
                        files={"file": ("silent.wav", wav, "audio/wav")})
        assert r.status_code == 400
        assert "silent" in r.json()["detail"].lower()

    def test_record_reject_short(self):
        wav = make_short_wav()
        r = client.post("/record", data={"treat_as": "clean"},
                        files={"file": ("short.wav", wav, "audio/wav")})
        assert r.status_code == 400
        assert "short" in r.json()["detail"].lower()

    def test_record_reject_non_wav(self):
        r = client.post("/record", data={"treat_as": "clean"},
                        files={"file": ("bad.txt", make_non_wav(), "text/plain")})
        assert r.status_code == 400

    def test_record_truncates_long(self):
        """A clip > 20 seconds should be truncated (not rejected)."""
        wav = make_wav_bytes(fs=16000, duration=25.0)
        r = client.post("/record", data={"treat_as": "clean"},
                        files={"file": ("long.wav", wav, "audio/wav")})
        assert r.status_code == 200
        p = r.json()
        # After truncation to 20s and resampling (same rate), duration <= 20
        assert p["duration_s"] <= 20.5


class TestFilter:
    def _get_audio_id(self):
        samples = client.get("/samples").json()["samples"]
        if not samples:
            return None
        r = client.post("/generate-noisy", json={
            "sample": samples[0], "snr_db": 5, "noise_start_hz": 3500
        })
        return r.json()["audio_id"]

    def test_filter_butterworth(self):
        aid = self._get_audio_id()
        if not aid:
            return
        r = client.post("/filter", json={
            "audio_id": aid, "filter_type": "butterworth",
            "order": 6, "cutoff_hz": 3400, "ripple_db": 1, "stopband_db": 40,
            "zero_phase": True
        })
        assert r.status_code == 200
        d = r.json()
        assert "filtered_audio" in d
        assert "frequency_response" in d
        assert "pole_zero" in d
        assert d["metrics"]["filter_type"] == "butterworth"

    def test_filter_on_recorded_clip(self):
        """Filter should work on a clip created via /record."""
        wav = make_wav_bytes(fs=48000, duration=3.0, channels=2)
        rr = client.post("/record", data={"treat_as": "clean", "snr_db": "5", "noise_start_hz": "3500"},
                         files={"file": ("rec.wav", wav, "audio/wav")})
        aid = rr.json()["audio_id"]
        r = client.post("/filter", json={
            "audio_id": aid, "filter_type": "butterworth",
            "order": 4, "cutoff_hz": 3400, "ripple_db": 1, "stopband_db": 40,
            "zero_phase": True
        })
        assert r.status_code == 200

    def test_filter_output_unchanged_shape(self):
        """Verify existing /filter output shape is unchanged."""
        aid = self._get_audio_id()
        if not aid:
            return
        r = client.post("/filter", json={
            "audio_id": aid, "filter_type": "chebyshev",
            "order": 4, "cutoff_hz": 3400, "ripple_db": 1, "stopband_db": 40,
            "zero_phase": False
        })
        d = r.json()
        expected_keys = {"fs", "filtered_audio", "filtered_waveform", "filtered_spectrum",
                         "frequency_response", "pole_zero", "metrics"}
        assert expected_keys.issubset(set(d.keys()))
        assert "f" in d["frequency_response"]
        assert "db" in d["frequency_response"]
        assert "zeros" in d["pole_zero"]
        assert "poles" in d["pole_zero"]


class TestOctaveStatus:
    def test_octave_status_returns(self):
        r = client.get("/octave-status")
        assert r.status_code == 200
        d = r.json()
        assert "available" in d
        assert "version" in d
        assert "reason" in d


class TestOctaveUnavailable:
    def test_octave_filter_503_when_unavailable(self):
        """If Octave is not installed, /octave-filter returns 503."""
        from octave_engine import engine
        if engine.available:
            return  # Can't test this if Octave is actually available
        wav = make_wav_bytes(fs=16000, duration=2.0)
        rr = client.post("/record", data={"treat_as": "clean", "snr_db": "5"},
                         files={"file": ("rec.wav", wav, "audio/wav")})
        aid = rr.json()["audio_id"]
        r = client.post("/octave-filter", json={
            "audio_id": aid, "filter_type": "butterworth",
            "order": 6, "cutoff_hz": 3400, "ripple_db": 1, "stopband_db": 40,
            "zero_phase": True
        })
        assert r.status_code == 503


class TestOctaveFilter:
    """Tests that run only when Octave is available."""

    def _skip_if_no_octave(self):
        from octave_engine import engine
        if not engine.available:
            print("  [SKIP] Octave not available")
            return True
        return False

    def _get_audio_id(self):
        samples = client.get("/samples").json()["samples"]
        if not samples:
            return None
        r = client.post("/generate-noisy", json={
            "sample": samples[0], "snr_db": 5, "noise_start_hz": 3500
        })
        return r.json()["audio_id"]

    def test_octave_filter_works(self):
        if self._skip_if_no_octave():
            return
        aid = self._get_audio_id()
        if not aid:
            return
        r = client.post("/octave-filter", json={
            "audio_id": aid, "filter_type": "butterworth",
            "order": 6, "cutoff_hz": 3400, "ripple_db": 1, "stopband_db": 40,
            "zero_phase": True
        })
        assert r.status_code == 200
        d = r.json()
        assert d["engine"] == "octave"
        assert "filtered_audio" in d

    def test_octave_filter_on_recorded_clip(self):
        if self._skip_if_no_octave():
            return
        wav = make_wav_bytes(fs=48000, duration=3.0, channels=2)
        rr = client.post("/record", data={"treat_as": "clean", "snr_db": "5"},
                         files={"file": ("rec.wav", wav, "audio/wav")})
        aid = rr.json()["audio_id"]
        r = client.post("/octave-filter", json={
            "audio_id": aid, "filter_type": "butterworth",
            "order": 4, "cutoff_hz": 3400, "ripple_db": 1, "stopband_db": 40,
            "zero_phase": True
        })
        assert r.status_code == 200

    def test_scipy_octave_agree(self):
        """Frequency responses should agree within 0.5 dB for all filter types."""
        if self._skip_if_no_octave():
            return
        aid = self._get_audio_id()
        if not aid:
            return

        for ftype in ["butterworth", "chebyshev", "elliptic"]:
            for order in [2, 6, 12]:
                req = {
                    "audio_id": aid, "filter_type": ftype,
                    "order": order, "cutoff_hz": 3400,
                    "ripple_db": 1, "stopband_db": 40,
                    "zero_phase": True,
                }
                sp = client.post("/filter", json=req).json()
                oc = client.post("/octave-filter", json=req).json()

                sp_db = np.array(sp["frequency_response"]["db"])
                oc_db = np.array(oc["frequency_response"]["db"])
                # Compare only where both are above -100 dB
                valid = (sp_db > -100) & (oc_db > -100)
                if valid.any():
                    max_diff = np.max(np.abs(sp_db[valid] - oc_db[valid]))
                    print(f"  {ftype} order {order}: max response diff = {max_diff:.4f} dB")
                    assert max_diff < 0.5, f"{ftype} order {order} diff {max_diff:.4f} dB > 0.5 dB"

    def test_high_order_stability(self):
        """Order 12 Chebyshev/Elliptic: all poles inside unit circle, no NaN/Inf."""
        if self._skip_if_no_octave():
            return
        aid = self._get_audio_id()
        if not aid:
            return

        for ftype in ["chebyshev", "elliptic"]:
            req = {
                "audio_id": aid, "filter_type": ftype,
                "order": 12, "cutoff_hz": 3400,
                "ripple_db": 1, "stopband_db": 40,
                "zero_phase": True,
            }
            for endpoint in ["/filter", "/octave-filter"]:
                r = client.post(endpoint, json=req)
                assert r.status_code == 200
                d = r.json()
                pz = d["pole_zero"]
                poles_re = np.array(pz["poles"]["re"])
                poles_im = np.array(pz["poles"]["im"])
                magnitudes = np.sqrt(poles_re**2 + poles_im**2)
                assert np.all(magnitudes < 1.0 + 1e-10), f"Poles outside unit circle for {ftype} {endpoint}"
                # Check filtered audio has no NaN or Inf
                # (we check the waveform data since filtered_audio is base64)
                wy = np.array(d["filtered_waveform"]["y"])
                assert np.all(np.isfinite(wy)), f"NaN/Inf in filtered waveform for {ftype} {endpoint}"


class TestCompare:
    def test_compare_503_when_no_octave(self):
        from octave_engine import engine
        if engine.available:
            return
        samples = client.get("/samples").json()["samples"]
        if not samples:
            return
        r = client.post("/generate-noisy", json={"sample": samples[0], "snr_db": 5, "noise_start_hz": 3500})
        aid = r.json()["audio_id"]
        r2 = client.post("/compare", json={
            "audio_id": aid, "filter_type": "butterworth",
            "order": 6, "cutoff_hz": 3400, "ripple_db": 1, "stopband_db": 40,
            "zero_phase": True
        })
        assert r2.status_code == 503

    def test_compare_works(self):
        from octave_engine import engine
        if not engine.available:
            print("  [SKIP] Octave not available")
            return
        samples = client.get("/samples").json()["samples"]
        if not samples:
            return
        r = client.post("/generate-noisy", json={"sample": samples[0], "snr_db": 5, "noise_start_hz": 3500})
        aid = r.json()["audio_id"]
        r2 = client.post("/compare", json={
            "audio_id": aid, "filter_type": "butterworth",
            "order": 6, "cutoff_hz": 3400, "ripple_db": 1, "stopband_db": 40,
            "zero_phase": True
        })
        assert r2.status_code == 200
        d = r2.json()
        assert "max_response_diff_db" in d
        assert "rms_signal_diff" in d
        assert "scipy_response" in d
        assert "octave_response" in d


class TestResample:
    def test_resample_to_16k(self):
        x = np.sin(2 * np.pi * 440 * np.arange(0, 2, 1/48000))
        y = dsp.resample_to_16k(x, 48000)
        expected_len = int(len(x) * 16000 / 48000)
        assert abs(len(y) - expected_len) < 5

    def test_resample_noop_at_16k(self):
        x = np.sin(2 * np.pi * 440 * np.arange(0, 1, 1/16000))
        y = dsp.resample_to_16k(x, 16000)
        assert y is x  # should return the same object


# ================================================================ runner
if __name__ == "__main__":
    passed = 0
    failed = 0
    skipped = 0
    total = 0

    for cls_name, cls in sorted(globals().items()):
        if not isinstance(cls, type) or not cls_name.startswith("Test") or cls_name == "TestClient":
            continue
        instance = cls()
        for method_name in sorted(dir(instance)):
            if not method_name.startswith("test_"):
                continue
            total += 1
            label = f"{cls_name}.{method_name}"
            try:
                print(f"  RUN  {label} ...", end=" ", flush=True)
                getattr(instance, method_name)()
                print("PASS")
                passed += 1
            except Exception as e:
                print(f"FAIL: {e}")
                failed += 1

    print(f"\n{'='*60}")
    print(f"  {passed} passed, {failed} failed, out of {total} tests")
    if failed:
        print("  SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("  ALL TESTS PASSED")
