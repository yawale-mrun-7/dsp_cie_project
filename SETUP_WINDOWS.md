# Windows Setup Guide

## 1. Install Python dependencies

```
cd backend
.\venv\Scripts\pip.exe install -r requirements.txt
```

## 2. Run the server

```
cd backend
.\venv\Scripts\uvicorn.exe main:app --reload
```

Open http://127.0.0.1:8000 in Chrome or Edge.

## 3. Allow microphone in Chrome/Edge for http://127.0.0.1:8000

Chrome and Edge treat `127.0.0.1` as a secure origin, so the microphone
permission prompt should appear automatically when you click "Record from
microphone". If it does not:

- **Chrome**: go to `chrome://settings/content/microphone`, add
  `http://127.0.0.1:8000` to the "Allowed" list.
- **Edge**: go to `edge://settings/content/microphone`, add
  `http://127.0.0.1:8000` to the "Allow" list.

**Important:** do not open `index.html` by double-clicking the file. Browsers
block microphone access on `file://` URLs. Always use `http://127.0.0.1:8000`.

## 4. Install GNU Octave (optional, for the Octave engine)

1. Download GNU Octave for Windows from https://octave.org/download
   (use the installer or the portable ZIP).
2. After installation, find the `octave-cli.exe` executable. Typical path:
   ```
   C:\Program Files\GNU Octave\Octave-9.2.0\mingw64\bin\octave-cli.exe
   ```
3. Set the `OCTAVE_EXECUTABLE` environment variable to point to it:
   ```
   set OCTAVE_EXECUTABLE=C:\Program Files\GNU Octave\Octave-9.2.0\mingw64\bin\octave-cli.exe
   ```
   Or in PowerShell:
   ```powershell
   $env:OCTAVE_EXECUTABLE = "C:\Program Files\GNU Octave\Octave-9.2.0\mingw64\bin\octave-cli.exe"
   ```
4. Install the Octave `signal` package. Open `octave-cli.exe` and run:
   ```
   pkg install -forge signal
   pkg load signal
   ```
   If it loads without errors, the package is ready.

## 5. Verify Oct2Py works

With `OCTAVE_EXECUTABLE` set, run these two lines in a Python shell:

```python
from oct2py import Oct2Py
oc = Oct2Py(); print(oc.eval("version()"))
```

If it prints the Octave version number (for example `9.2.0`), Oct2Py is
working. The server will detect Octave automatically on startup.

## 6. Run tests

```
cd backend
.\venv\Scripts\python.exe test_backend.py
```

Tests that need Octave will print `[SKIP]` if it is not available. All
other tests run without Octave.

## 7. Manual microphone checklist

| Step | Expected result |
|------|----------------|
| Click "Record from microphone", allow permission | Timer and level meter appear, button says "Stop" |
| Deny microphone permission | Red alert: "Microphone permission was denied..." |
| Record with no microphone connected | Red alert: "No microphone was found..." |
| Click "Stop" before 1 second | Red alert: "Recording is too short..." |
| Let recording run to 15 seconds | Auto-stops, preview appears |
| Play preview, click "Use this recording" with "clean" radio | Noisy clip loads, SNR numbers shown, charts fill in |
| Switch to "already has noise" radio and re-record | SNR shows n/a, charts work |
| After recording, move filter sliders | Charts and audio update live |
| Switch Engine to Octave (if available) | Charts update using Octave engine |
| Click "Compare engines" (if Octave available) | Compare card shows overlaid responses and diff numbers |
