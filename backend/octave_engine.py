"""
octave_engine.py: wrapper around Oct2Py for the Octave IIR filter engine.

One Oct2Py instance is created at application startup (FastAPI lifespan).
All calls go through a threading.Lock because oct2py is not thread-safe and
FastAPI's synchronous endpoints may run concurrently via a thread pool.

If Octave is not installed, or the signal package is missing, or the
executable cannot be found, the wrapper records the reason and marks itself
unavailable. The rest of the server starts normally in that case.

On Windows, set the environment variable OCTAVE_EXECUTABLE to the full path
of octave-cli.exe, for example:
    set OCTAVE_EXECUTABLE=C:\\Program Files\\GNU Octave\\Octave-9.2.0\\mingw64\\bin\\octave-cli.exe
"""

import logging
import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

OCTAVE_DIR = Path(__file__).parent / "octave"


class OctaveEngine:
    """Thin, thread-safe wrapper around a single oct2py.Oct2Py instance."""

    def __init__(self):
        self._lock = threading.Lock()
        self._oc = None          # Oct2Py instance, or None when unavailable
        self.available = False
        self.version: Optional[str] = None
        self.reason: Optional[str] = None
        self._try_init()

    # ---------------------------------------------------------------- startup
    def _try_init(self):
        try:
            import oct2py  # noqa: F401 -- imported for version check first
        except ImportError:
            self.reason = "oct2py is not installed (pip install oct2py)."
            log.warning("Octave engine: %s", self.reason)
            return

        exe = os.environ.get("OCTAVE_EXECUTABLE", "")
        kwargs = {}
        if exe:
            kwargs["executable"] = exe

        try:
            from oct2py import Oct2Py
            oc = Oct2Py(**kwargs)
            # Add the octave/ folder so iir_lowpass.m is on the path
            oc.addpath(str(OCTAVE_DIR))
            # Smoke-test: load the signal package and run a trivial call
            oc.eval("pkg load signal;")
            ver_str = oc.eval("version();", verbose=False)
            self._oc = oc
            self.available = True
            self.version = str(ver_str).strip()
            log.info("Octave engine ready (version %s).", self.version)
        except Exception as exc:
            self.reason = str(exc)
            log.warning("Octave engine unavailable: %s", exc)

    # ---------------------------------------------------------------- public API
    def status(self) -> dict:
        return {
            "available": self.available,
            "version": self.version,
            "reason": self.reason,
        }

    def run_filter(
        self,
        x: np.ndarray,
        fs: int,
        ftype: str,
        order: int,
        cutoff_hz: float,
        ripple_db: float,
        stopband_db: float,
        zero_phase: bool,
    ) -> dict:
        """
        Call iir_lowpass.m and return a dict with the same keys as dsp.py
        helper functions produce, so main.py can assemble the /octave-filter
        response without any special casing.

        Returns:
            {
              "y":       np.ndarray  - filtered signal
              "sos":     np.ndarray  - Nx6 SOS matrix
              "zeros":   dict{re, im}
              "poles":   dict{re, im}
              "freq_hz": list[float]
              "freq_db": list[float]
            }
        """
        if not self.available:
            raise RuntimeError("Octave is not available: " + (self.reason or "unknown reason"))

        # Octave functions expect a column vector
        x_col = x.reshape(-1, 1).astype(np.float64)

        with self._lock:
            try:
                y, sos, z, p, w, h_db = self._oc.iir_lowpass(
                    x_col,
                    float(fs),
                    ftype,
                    float(order),
                    float(cutoff_hz),
                    float(ripple_db),
                    float(stopband_db),
                    bool(zero_phase),
                    nout=6,
                )
            except Exception as exc:
                raise RuntimeError(f"Octave call failed: {exc}") from exc

        # oct2py may return 2-D column arrays; flatten to 1-D
        y = np.array(y, dtype=np.float64).ravel()
        w = np.array(w, dtype=np.float64).ravel()
        h_db_arr = np.array(h_db, dtype=np.float64).ravel()

        # Poles and zeros come back as complex arrays (may be 2-D)
        z_arr = np.array(z, dtype=complex).ravel()
        p_arr = np.array(p, dtype=complex).ravel()

        return {
            "y": y,
            "sos": np.array(sos, dtype=np.float64),
            "zeros": {"re": np.real(z_arr).tolist(), "im": np.imag(z_arr).tolist()},
            "poles": {"re": np.real(p_arr).tolist(), "im": np.imag(p_arr).tolist()},
            "freq_hz": w.tolist(),
            "freq_db": np.maximum(h_db_arr, -120.0).tolist(),
        }


# Singleton -- created once when this module is imported.
# main.py imports `engine` and calls engine.status() / engine.run_filter().
engine = OctaveEngine()
