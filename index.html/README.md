# Sheriff Robot View

A Python/OpenCV webcam project that detects a face and estimates its approximate XYZ position.

## Features

- Real-time face detection using OpenCV YuNet
- Face bounding box and center marker
- Approximate X (left/right), Y (up/down), and Z (distance) display
- Basic smoothing to reduce coordinate jitter

## Requirements

- Python 3.14 (or a compatible Python version)
- A webcam
- Internet connection on first run to download the YuNet model

## Setup

Open a terminal in this folder and run:

```powershell
python -m pip install -r requirements.txt
python step5.py
```

The model is downloaded automatically on first launch and saved next to `step5.py`.

Press `Q` or `Esc` to exit.

## Coordinate notes

- X: horizontal position relative to the camera image center; positive is right.
- Y: vertical position relative to the camera image center; positive is down.
- Z: approximate distance in centimeters inferred from apparent face width.

**Important:** Z is only an estimate. The code uses assumed face width and focal factor, and does not perform camera calibration. Accuracy varies by camera, resolution, person, and distance. Do not use these values for safety-critical purposes.

## Files

- `step5.py` — main program
- `requirements.txt` — Python dependencies
- `.gitignore` — ignores generated files and the downloaded model
