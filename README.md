# DriveSafe - Lane Detection System

A real-time lane detection system using OpenCV and PyQt5 that processes live camera feed to identify and highlight road lanes.

## Features

- Real-time camera feed processing
- Lane detection using OpenCV
- Side-by-side view of raw and processed feed
- Simple and intuitive GUI interface

## Requirements

- Python 3.7+
- OpenCV
- PyQt5
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ojaskandy/drivesafe.git
cd drivesafe
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python lane_detection_gui.py
```

The application will:
- Open your default camera
- Show the raw feed on the left
- Show the processed feed with lane detection on the right
- Update in real-time
- Close properly when you close the window

## How It Works

The application uses:
- OpenCV for image processing and lane detection
- PyQt5 for the GUI interface
- Hough Transform for line detection
- Region of Interest masking for focused processing

## License

MIT License 