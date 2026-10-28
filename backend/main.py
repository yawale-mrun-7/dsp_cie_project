"""
main.py: FastAPI backend for the IIR low pass voice denoiser.

Run:  uvicorn main:app --reload
App:  http://127.0.0.1:8000
Docs: http://127.0.0.1:8000/docs
"""
import io
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import numpy as np
import scipy
from scipy.io import wavfile

import dsp
from octave_engine import engine as octave_engine

log = logging.getLogger(__name__)

app = FastAPI(title="IIR Low Pass Voice Denoiser")

# Allow the frontend (opened from a file or another local port) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLES_DIR = Path(__file__).parent / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)
STATIC_DIR = Path(__file__).parent / "static"        # holds index.html
RECORDINGS_DIR = Path(__file__).parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

# Tiny in-memory store so /filter can refer to audio by id instead of re-uploading it.
# No database needed for this project; old entries are dropped.
STORE: dict = {}
MAX_STORE = 20

# Silence threshold: RMS below this fraction of full scale is considered silent.
SILENCE_RMS = 1e-4


# ---------------------------------------------------------------- store helpers
def _save(fs, clean, noisy) -> str:
    audio_id = uuid.uuid4().hex[:12]
    STORE[audio_id] = {"fs": fs, "clean": clean, "noisy": noisy}
    while len(STORE) > MAX_STORE:
        STORE.pop(next(iter(STORE)))
    return audio_id


def _noisy_payload(audio_id: str) -> dict:
    e = STORE[audio_id]
    fs, clean, noisy = e["fs"], e["clean"], e["noisy"]
    t, y = dsp.waveform(noisy, fs)
    f, db = dsp.spectrum(noisy, fs)
    return {
        "audio_id": audio_id,
        "fs": fs,
        "duration_s": round(len(noisy) / fs, 2),
        "has_reference": clean is not None,
        "noisy_audio": dsp.to_wav_base64(noisy, fs),
        "noisy_waveform": {"t": t, "y": y},
        "noisy_spectrum": {"f": f, "db": db},
        "snr_noisy_db": dsp.snr_db(clean, noisy) if clean is not None else None,
    }


# ---------------------------------------------------------------- request models
class NoisyRequest(BaseModel):
    sample: str = Field(description="File name (without .wav) from GET /samples")
    snr_db: float = Field(5.0, ge=-5, le=30, description="Target SNR of the noisy clip")
    noise_start_hz: float = Field(3500.0, ge=500, le=20000, description="Hiss only exists above this frequency")


class FilterRequest(BaseModel):
    audio_id: str
    filter_type: Literal["butterworth", "chebyshev", "elliptic"] = "butterworth"
    order: int = Field(6, ge=1, le=12)
    cutoff_hz: float = Field(3400.0, gt=0)
    ripple_db: float = Field(1.0, gt=0, le=5, description="Passband ripple (Chebyshev, Elliptic)")
    stopband_db: float = Field(40.0, ge=10, le=120, description="Stopband attenuation (Elliptic)")
    zero_phase: bool = Field(True, description="Filter forward and backward (no phase delay). False = ordinary causal IIR")


# ---------------------------------------------------------------- endpoints
@app.get("/", include_in_schema=False)
def home():
    """Serve the frontend, so one command runs the whole project."""
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(404, "static/index.html not found. Put the frontend file in backend/static/.")
    return FileResponse(page)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/samples")
def list_samples():
    return {"samples": sorted(p.stem for p in SAMPLES_DIR.glob("*.wav"))}


@app.post("/generate-noisy")
def generate_noisy(req: NoisyRequest):
    """Take a clean sample from the samples folder, add hiss, and return the noisy clip."""
    available = {p.stem for p in SAMPLES_DIR.glob("*.wav")}
    if req.sample not in available:
        raise HTTPException(404, f"Unknown sample '{req.sample}'. Available: {sorted(available)}")

    fs, clean = dsp.read_wav((SAMPLES_DIR / f"{req.sample}.wav").read_bytes())
    noisy = dsp.add_hiss(clean, fs, req.snr_db, req.noise_start_hz)
    return _noisy_payload(_save(fs, clean, noisy))


@app.post("/upload")
def upload(
    file: UploadFile = File(...),
    treat_as: Literal["clean", "noisy"] = Form("noisy"),
    snr_db: float = Form(5.0),
    noise_start_hz: float = Form(3500.0),
):
    """
    Upload a WAV file.
    treat_as="clean": The file is saved to samples/ as a clean reference, resampled to
      16000 Hz, and simulated hiss is added according to snr_db and noise_start_hz.
      The user can adjust sliders later to re-generate hiss, and full SNR metrics are displayed.
    treat_as="noisy": The clip is treated as already noisy with no clean reference.
    """
    try:
        raw = file.file.read()
        fs_orig, x = dsp.read_wav(raw)
    except Exception:
        raise HTTPException(400, "Could not read that file. Please upload a PCM WAV file.")
    if len(x) < 2048:
        raise HTTPException(400, "Clip is too short.")

    if treat_as == "clean":
        x16 = dsp.resample_to_16k(x, fs_orig)
        fs = dsp.TARGET_FS
        clean = x16
        noisy = dsp.add_hiss(clean, fs, snr_db, noise_start_hz)

        # Save to SAMPLES_DIR so it appears in the dropdown and can be re-used with the sliders
        base_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", Path(file.filename or "upload").stem)
        sample_name = f"user_{base_name}_{uuid.uuid4().hex[:4]}"
        try:
            pcm = (np.clip(clean, -1.0, 1.0) * 32767).astype(np.int16)
            buf = io.BytesIO()
            wavfile.write(buf, fs, pcm)
            (SAMPLES_DIR / f"{sample_name}.wav").write_bytes(buf.getvalue())
        except Exception as exc:
            log.warning("Could not persist clean sample to disk: %s", exc)

        audio_id = _save(fs, clean, noisy)
        payload = _noisy_payload(audio_id)
        payload["sample_name"] = sample_name
        return payload
    else:
        # treat_as == "noisy"
        audio_id = _save(fs_orig, None, x)
        return _noisy_payload(audio_id)


@app.post("/record")
def record(
    file: UploadFile = File(...),
    treat_as: Literal["clean", "noisy"] = Form("clean"),
    snr_db: float = Form(5.0),
    noise_start_hz: float = Form(3500.0),
):
    """
    Accept a browser-recorded WAV (16-bit PCM), resample to 16000 Hz, validate
    it, and return the same payload shape as /generate-noisy so the frontend
    reuses its existing handler.

    treat_as="clean": hiss is added and the original is kept as the clean
      reference, so SNR improvement numbers are available.
    treat_as="noisy": the recording is treated as already noisy; no clean
      reference is stored, so SNR shows n/a.
    """
    try:
        raw = file.file.read()
        fs_orig, x = dsp.read_wav(raw)
    except Exception:
        raise HTTPException(400, "Could not read the uploaded file. Please upload a PCM WAV.")

    # Reject clips shorter than 1 second (checked before resampling)
    if len(x) / fs_orig < 1.0:
        raise HTTPException(400, "Recording is too short (minimum 1 second). Please try again.")

    # Reject near-silent recordings (RMS below threshold, before resampling)
    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms < SILENCE_RMS:
        raise HTTPException(
            400,
            "The recording is silent. Check that the right microphone is selected."
        )

    # Resample to 16000 Hz so every recording has the same 0-8 kHz spectrum
    x16 = dsp.resample_to_16k(x, fs_orig)
    fs = dsp.TARGET_FS

    # Optional: save the resampled clean recording to disk for later offline analysis
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rec_name = f"rec_{ts}_{uuid.uuid4().hex[:6]}.wav"
    sample_name = None
    try:
        pcm = (np.clip(x16, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        wavfile.write(buf, fs, pcm)
        (RECORDINGS_DIR / rec_name).write_bytes(buf.getvalue())

        # If clean, also expose as a sample so sliders can be adjusted later
        if treat_as == "clean":
            sample_name = f"rec_{ts}_{uuid.uuid4().hex[:4]}"
            (SAMPLES_DIR / f"{sample_name}.wav").write_bytes(buf.getvalue())
    except Exception as exc:
        log.warning("Could not save recording to disk: %s", exc)
        rec_name = None

    if treat_as == "clean":
        clean = x16
        noisy = dsp.add_hiss(clean, fs, snr_db, noise_start_hz)
        audio_id = _save(fs, clean, noisy)
    else:
        noisy = x16
        audio_id = _save(fs, None, noisy)

    payload = _noisy_payload(audio_id)
    if rec_name:
        payload["saved_recording"] = rec_name
    if sample_name:
        payload["sample_name"] = sample_name
    return payload


@app.post("/filter")
def filter_audio(req: FilterRequest):
    """Design the chosen IIR low pass filter, apply it, and return audio, plots data and metrics."""
    entry = STORE.get(req.audio_id)
    if entry is None:
        raise HTTPException(404, "Unknown audio_id. Generate or upload a clip first.")
    fs, clean, noisy = entry["fs"], entry["clean"], entry["noisy"]

    t0 = time.perf_counter()
    try:
        sos = dsp.design_sos(
            req.filter_type, req.order, req.cutoff_hz, fs, req.ripple_db, req.stopband_db
        )
    except ValueError as err:
        raise HTTPException(400, str(err))

    filtered = dsp.apply_filter(sos, noisy, req.zero_phase)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    freqs, mag_db = dsp.frequency_response(sos, fs)
    zeros, poles = dsp.pole_zero(sos)
    f, db = dsp.spectrum(filtered, fs)
    t, y = dsp.waveform(filtered, fs)

    snr_before: Optional[float] = None
    snr_after: Optional[float] = None
    if clean is not None:
        snr_before = dsp.snr_db(clean, noisy)
        snr_after = dsp.snr_db(clean, filtered)

    return {
        "engine": "scipy",
        "engine_info": {
            "name": "SciPy",
            "version": scipy.__version__,
            "method": "scipy.signal.sosfiltfilt" if req.zero_phase else "scipy.signal.sosfilt",
            "elapsed_ms": elapsed_ms,
        },
        "fs": fs,
        "filtered_audio": dsp.to_wav_base64(filtered, fs),
        "filtered_waveform": {"t": t, "y": y},
        "filtered_spectrum": {"f": f, "db": db},
        "frequency_response": {"f": freqs, "db": mag_db},
        "pole_zero": {"zeros": zeros, "poles": poles},
        "metrics": {
            "filter_type": req.filter_type,
            "order": req.order,
            "cutoff_hz": req.cutoff_hz,
            "snr_before_db": snr_before,
            "snr_after_db": snr_after,
            "snr_improvement_db": (
                snr_after - snr_before if snr_before is not None and snr_after is not None else None
            ),
        },
    }


# ---------------------------------------------------------------- Octave endpoints
@app.get("/octave-status")
def octave_status():
    """Return whether the Octave engine is available."""
    return octave_engine.status()


@app.post("/octave-filter")
def octave_filter(req: FilterRequest):
    """
    Same contract as /filter but uses the Octave IIR engine.
    Returns HTTP 503 when Octave is unavailable.
    """
    if not octave_engine.available:
        raise HTTPException(
            503,
            "Octave engine is not available: " + (octave_engine.reason or "unknown reason"),
        )

    entry = STORE.get(req.audio_id)
    if entry is None:
        raise HTTPException(404, "Unknown audio_id. Generate or upload a clip first.")
    fs, clean, noisy = entry["fs"], entry["clean"], entry["noisy"]

    # Validate parameters the same way /filter does, before calling Octave
    if not 0 < req.cutoff_hz < fs / 2:
        raise HTTPException(400, f"Cutoff must be between 0 and {fs / 2:.0f} Hz for this clip.")
    if not 1 <= req.order <= 12:
        raise HTTPException(400, "Order must be between 1 and 12.")

    t0 = time.perf_counter()
    try:
        result = octave_engine.run_filter(
            noisy, fs,
            req.filter_type, req.order, req.cutoff_hz,
            req.ripple_db, req.stopband_db, req.zero_phase,
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    total_ms = round((time.perf_counter() - t0) * 1000, 2)

    filtered = result["y"]
    f_spec, db_spec = dsp.spectrum(filtered, fs)
    t_wave, y_wave = dsp.waveform(filtered, fs)

    snr_before: Optional[float] = None
    snr_after: Optional[float] = None
    if clean is not None:
        snr_before = dsp.snr_db(clean, noisy)
        snr_after = dsp.snr_db(clean, filtered)

    return {
        "engine": "octave",
        "engine_info": {
            "name": "GNU Octave",
            "version": octave_engine.version,
            "executable": octave_engine.executable,
            "script": "octave/iir_lowpass.m",
            "elapsed_ms": result.get("elapsed_ms", total_ms),
            "command": f"iir_lowpass(x, {fs}, '{req.filter_type}', {req.order}, {req.cutoff_hz:.0f}, {req.ripple_db:.1f}, {req.stopband_db:.1f}, {str(req.zero_phase).lower()})",
        },
        "fs": fs,
        "filtered_audio": dsp.to_wav_base64(filtered, fs),
        "filtered_waveform": {"t": t_wave, "y": y_wave},
        "filtered_spectrum": {"f": f_spec, "db": db_spec},
        "frequency_response": {"f": result["freq_hz"], "db": result["freq_db"]},
        "pole_zero": {"zeros": result["zeros"], "poles": result["poles"]},
        "metrics": {
            "filter_type": req.filter_type,
            "order": req.order,
            "cutoff_hz": req.cutoff_hz,
            "snr_before_db": snr_before,
            "snr_after_db": snr_after,
            "snr_improvement_db": (
                snr_after - snr_before if snr_before is not None and snr_after is not None else None
            ),
        },
    }


@app.post("/compare")
def compare(req: FilterRequest):
    """
    Run both engines with the same request and return a comparison report.

    Returns HTTP 503 if Octave is unavailable.
    """
    if not octave_engine.available:
        raise HTTPException(
            503,
            "Octave engine is not available: " + (octave_engine.reason or "unknown reason"),
        )

    entry = STORE.get(req.audio_id)
    if entry is None:
        raise HTTPException(404, "Unknown audio_id.")
    fs, clean, noisy = entry["fs"], entry["clean"], entry["noisy"]

    if not 0 < req.cutoff_hz < fs / 2:
        raise HTTPException(400, f"Cutoff must be between 0 and {fs / 2:.0f} Hz for this clip.")

    # SciPy engine
    t0_sp = time.perf_counter()
    try:
        sos = dsp.design_sos(req.filter_type, req.order, req.cutoff_hz, fs, req.ripple_db, req.stopband_db)
    except ValueError as err:
        raise HTTPException(400, str(err))
    sp_filtered = dsp.apply_filter(sos, noisy, req.zero_phase)
    sp_elapsed_ms = round((time.perf_counter() - t0_sp) * 1000, 2)
    sp_freqs, sp_db = dsp.frequency_response(sos, fs)

    # Octave engine
    t0_oct = time.perf_counter()
    try:
        oct_result = octave_engine.run_filter(
            noisy, fs,
            req.filter_type, req.order, req.cutoff_hz,
            req.ripple_db, req.stopband_db, req.zero_phase,
        )
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    oct_elapsed_ms = oct_result.get("elapsed_ms", round((time.perf_counter() - t0_oct) * 1000, 2))
    oct_filtered = oct_result["y"]
    oct_freqs = oct_result["freq_hz"]
    oct_db = oct_result["freq_db"]

    import numpy as np

    # Align lengths in case of floating-point resampling differences
    min_len = min(len(sp_filtered), len(oct_filtered))
    sp_filtered = sp_filtered[:min_len]
    oct_filtered = oct_filtered[:min_len]
    noisy_trim = noisy[:min_len]

    # Frequency response difference (ignore points below -100 dB in both)
    sp_db_arr = np.array(sp_db)
    oct_db_arr = np.array(oct_db)
    valid = (sp_db_arr > -100) & (oct_db_arr > -100)
    if valid.any():
        max_resp_diff_db = float(np.max(np.abs(sp_db_arr[valid] - oct_db_arr[valid])))
    else:
        max_resp_diff_db = None

    # SNR improvement difference
    snr_diff = None
    if clean is not None:
        sp_snr_after = dsp.snr_db(clean, sp_filtered)
        oct_snr_after = dsp.snr_db(clean, oct_filtered)
        snr_before = dsp.snr_db(clean, noisy)
        if sp_snr_after is not None and oct_snr_after is not None and snr_before is not None:
            snr_diff = (sp_snr_after - snr_before) - (oct_snr_after - snr_before)

    # Signal differences (numerical variation & transient differences)
    diff_signal = sp_filtered - oct_filtered
    rms_diff = float(np.sqrt(np.mean(diff_signal ** 2)))
    max_signal_diff = float(np.max(np.abs(diff_signal)))

    # Difference waveform downsampled for plotting
    t_diff, y_diff = dsp.waveform(diff_signal, fs)

    return {
        "max_response_diff_db": max_resp_diff_db,
        "snr_improvement_diff_db": snr_diff,
        "rms_signal_diff": rms_diff,
        "max_signal_diff": max_signal_diff,
        "scipy_elapsed_ms": sp_elapsed_ms,
        "octave_elapsed_ms": oct_elapsed_ms,
        "scipy_version": scipy.__version__,
        "octave_version": octave_engine.version,
        "octave_executable": octave_engine.executable,
        "difference_waveform": {"t": t_diff, "y": y_diff},
        "scipy_response": {"f": sp_freqs, "db": sp_db},
        "octave_response": {"f": oct_freqs, "db": oct_db},
    }
