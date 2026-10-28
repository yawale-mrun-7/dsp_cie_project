# IIR Low-Pass Voice Denoiser

A Digital Signal Processing (DSP) project that demonstrates voice-noise reduction using Infinite Impulse Response (IIR) low-pass filters. The application provides a browser-based interface backed by a Python FastAPI server and supports Butterworth, Chebyshev Type I, and Elliptic filter designs.

## Features

- Record voice directly from a supported browser microphone.
- Upload a PCM WAV audio file for processing.
- Generate a noisy test signal by adding high-frequency hiss to a clean sample.
- Adjust the target signal-to-noise ratio (SNR) and the frequency at which simulated hiss begins.
- Apply three IIR low-pass filter types:
  - **Butterworth** — maximally flat passband.
  - **Chebyshev Type I** — sharper transition with passband ripple.
  - **Elliptic** — sharp transition with ripple in both passband and stopband.
- Configure filter order, cutoff frequency, passband ripple, stopband attenuation, and zero-phase filtering where supported.
- Inspect waveforms, frequency spectra, filter frequency response, and pole-zero information where shown in the interface.
- View SNR measurements when a clean reference signal is available.
- Optionally use GNU Octave as a second filtering engine and compare its results with the Python/SciPy engine.
- Access interactive API documentation through FastAPI.

## Technology Stack

- **Backend:** Python, FastAPI, Uvicorn
- **DSP and numerical processing:** NumPy, SciPy
- **Frontend:** HTML, CSS, JavaScript
- **Audio input:** Browser microphone and WAV file upload
- **Optional second engine:** GNU Octave, Oct2Py, and Octave's `signal` package

## Project Structure

```text
dsp_cie_project_new_updates/
├── backend/
│   ├── main.py                  # FastAPI application and API endpoints
│   ├── dsp.py                   # Audio processing, IIR filters, plots, and metrics
│   ├── octave_engine.py         # Optional Octave integration
│   ├── make_synthetic_sample.py # Creates a synthetic test sample
│   ├── requirements.txt         # Python dependencies
│   ├── octave/
│   │   └── iir_lowpass.m        # Octave filter implementation
│   ├── samples/                 # Test WAV files
│   ├── recordings/              # Saved recordings, if created
│   └── static/
│       └── index.html           # Web interface served by FastAPI
└── SETUP_WINDOWS.md             # Additional Windows setup notes
```

> Run commands from the `backend` directory unless stated otherwise.

## Requirements

- Windows, macOS, or Linux
- Python 3.10+ recommended
- A modern browser such as Google Chrome or Microsoft Edge
- GNU Octave is optional and only needed for the Octave engine and engine-comparison functionality

## Setup and Run on Windows

### 1. Open the project folder

Open Command Prompt and navigate to the project’s backend folder. Update the path if your project is stored elsewhere.

```cmd
cd /d D:\dsp_project\dsp_cie_project_new_updates\backend
```

### 2. Create a virtual environment

```cmd
py -m venv venv
```

Activate it:

```cmd
venv\Scripts\activate.bat
```

If you already created and activated this environment, you can skip this step.

### 3. Install dependencies

```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Start the server

```cmd
python -m uvicorn main:app --reload
```

When the server starts successfully, open:

- **Application:** http://127.0.0.1:8000
- **API documentation:** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health

Keep the terminal running while using the application. To stop the server, press `Ctrl+C`.

### 5. Allow microphone access

Open the application at `http://127.0.0.1:8000` rather than opening the HTML file directly. When prompted, allow microphone access. If no prompt appears, check your browser's microphone permissions for `http://127.0.0.1:8000`.

## Basic Usage

1. Open the application in your browser.
2. Record a short voice sample or upload a supported PCM WAV file.
3. If the recording is clean, choose the clean-reference option so the application can add simulated hiss and calculate reference-based SNR. If the recording already contains noise, choose the already-noisy option.
4. Select an IIR filter type and adjust its parameters.
5. Apply the filter and compare the input and filtered audio, waveforms, spectra, and available metrics.
6. Repeat with the other filter types while keeping the input and settings consistent for a fair comparison.

### Suggested starting settings

- Filter: Butterworth
- Order: 6
- Cutoff frequency: 3,400 Hz
- Zero-phase filtering: enabled, if available

These are example starting values, not universally optimal settings. The cutoff must be below the Nyquist frequency (half the sample rate). A low-pass filter attenuates frequencies above the cutoff, so it may also remove high-frequency speech components and will not eliminate noise that overlaps the speech spectrum.

## Optional GNU Octave Engine

The application can use Python/SciPy without GNU Octave. To enable the optional Octave engine:

1. Install GNU Octave from https://octave.org/download.
2. Install or enable Octave's `signal` package.
3. Make sure `octave-cli.exe` is discoverable. If automatic detection fails, set `OCTAVE_EXECUTABLE` to its full path in the same terminal before starting the backend.

Example for Command Prompt (adjust the version/path to match your installation):

```cmd
set OCTAVE_EXECUTABLE=C:\Program Files\GNU Octave\Octave-9.2.0\mingw64\bin\octave-cli.exe
```

Then start the server from that same terminal. Octave-specific functions will remain unavailable if Octave or its `signal` package cannot be initialized.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Serves the web interface |
| `GET` | `/health` | Checks backend status |
| `GET` | `/samples` | Lists available WAV test samples |
| `POST` | `/generate-noisy` | Adds simulated hiss to a clean sample |
| `POST` | `/upload` | Uploads a WAV file |
| `POST` | `/filter` | Filters a loaded audio clip |
| `/docs` | Interactive API docs | Explore and test API endpoints |

The API documentation at `/docs` is the source of truth for the current endpoint parameters and response schemas.

## DSP Concepts Demonstrated

- Sampling frequency and the Nyquist limit
- Digital IIR low-pass filter design
- Butterworth, Chebyshev Type I, and Elliptic filter characteristics
- Filter order, cutoff frequency, passband ripple, and stopband attenuation
- Causal filtering versus forward-backward (zero-phase) filtering
- Time-domain waveform and frequency-domain spectrum analysis
- Signal-to-noise ratio (SNR) as an evaluation metric
- Pole-zero representation of digital filters
- Comparison of Python/SciPy and GNU Octave implementations

## Troubleshooting

- **`py` or `python` is not recognized:** Install Python and reopen the terminal.
- **`requirements.txt` not found:** Confirm that the terminal is inside the `backend` directory.
- **`static/index.html` not found:** Confirm that `backend/static/index.html` exists.
- **Port 8000 is already in use:** Stop the other server using that port, or start this instance on another port with `python -m uvicorn main:app --reload --port 8001`.
- **Microphone does not work:** Use the application through `http://127.0.0.1:8000`, allow microphone access in browser settings, and check that a microphone is connected.
- **WAV upload fails:** Use a valid PCM WAV file.
- **Octave engine is unavailable:** This is optional; verify the Octave installation, executable path, and `signal` package if you need it.

## Notes

- SNR values are most meaningful when a clean reference signal is available. For an already-noisy upload without a clean reference, reference-based SNR may be unavailable.
- Filter settings should be compared using the same input audio and consistent evaluation conditions.
- Audio files saved during testing may contain personal voice recordings. Review the contents of `backend/recordings/` before sharing the project publicly.

## License

No license is specified in this repository. Add a license file if you intend to distribute or reuse the project under explicit terms.
