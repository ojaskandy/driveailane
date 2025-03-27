# Traffic Light Detection System

This application uses a specialized YOLOv8 model trained for traffic light detection and color classification. The system can detect and classify traffic lights as red, yellow, or green in real-time using your webcam or from video files.

## Required Files
1. `traffic_light_detector.py` - Main application file
2. `best_traffic_small_yolo.pt` - YOLOv8 model file (download from [Traffic-Light-Detection-Color-Classification](https://github.com/Syazvinski/Traffic-Light-Detection-Color-Classification))
3. `requirements.txt` - Python dependencies

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the model file:
- Get `best_traffic_small_yolo.pt` from the [original repository](https://github.com/Syazvinski/Traffic-Light-Detection-Color-Classification)
- Place it in the same directory as `traffic_light_detector.py`

## Usage

Run the application:
```bash
python traffic_light_detector.py
```

### Features
- Real-time traffic light detection
- Color classification (Red, Yellow, Green)
- Live statistics
- Detection history
- Save detected frames automatically
- Support for both webcam and video file input

### Model Performance
| Model Size | Execution Time(MS) | Hardware Used         |
|------------|-------------------|---------------------|
| Small      | 72.1             | Mac M1 Max (CPU Only) |

## Directory Structure
```
.
├── README.md
├── requirements.txt
├── traffic_light_detector.py
├── best_traffic_small_yolo.pt
└── detected_traffic_lights/
    └── (Detection results will be saved here)
```

## Development

This project is built on top of the traffic lights detection system from [farukalamai/traffic-lights-detection-and-color-recognition-using-yolov8](https://github.com/farukalamai/traffic-lights-detection-and-color-recognition-using-yolov8).

## License

[License information will be added]

## Contributing

[Contribution guidelines will be added] 