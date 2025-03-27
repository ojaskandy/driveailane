# Lane Detection GUI Application

A real-time lane detection application that uses computer vision to detect and highlight lanes in a video feed. The application provides a side-by-side view of the raw camera feed and the processed view with detected lanes.

## Features

- Real-time camera feed processing
- Lane detection using OpenCV
- Side-by-side view of raw and processed feeds
- Simple and intuitive GUI interface

## Requirements

- Python 3.x
- OpenCV
- NumPy
- PyQt5
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ojaskandy/driveailane.git
cd driveailane
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python3 lane_detection_gui.py
```

The application will open a window showing:
- Left side: Raw camera feed
- Right side: Processed view with detected lanes

## How it Works

The application uses the following steps for lane detection:
1. Captures video feed from the camera
2. Converts the frame to grayscale
3. Applies Canny edge detection
4. Defines a region of interest
5. Uses Hough transform to detect lines
6. Draws the detected lines on the processed view

## License

This project is open source and available under the MIT License. 