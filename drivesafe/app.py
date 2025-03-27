from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
from datetime import datetime
import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from torch.serialization import add_safe_globals

# Add necessary classes to safe globals since we trust our model
safe_classes = [
    DetectionModel,
    nn.Sequential,
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.ReLU,
    nn.Module,
    nn.ModuleList
]
for cls in safe_classes:
    add_safe_globals([cls])

def load_trusted_model(model_path):
    """Load a trusted YOLO model with weights_only=False"""
    # Monkey patch torch.load to use weights_only=False
    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load
    
    try:
        # Load the model
        model = YOLO(model_path)
        return model
    finally:
        # Restore original torch.load
        torch.load = original_torch_load

app = Flask(__name__)

class DriveSafeDetector:
    def __init__(self):
        self.setup_models()
        self.setup_directories()
        self.video = None
        self.is_running = False
        self.camera_index = 0  # Default camera index
        
        # Frame dimensions
        self.frameWidth = 640
        self.frameHeight = 480
        
    def setup_directories(self):
        """Create necessary directories for outputs"""
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        self.traffic_light_output_dir = os.path.join(self.output_dir, "traffic_light_detections")
        self.lane_output_dir = os.path.join(self.output_dir, "lane_detections")
        os.makedirs(self.traffic_light_output_dir, exist_ok=True)
        os.makedirs(self.lane_output_dir, exist_ok=True)
        
    def setup_models(self):
        """Load detection models"""
        # Traffic light model path
        traffic_model_path = os.path.join("models", "traffic_light", "best_traffic_small_yolo.pt")
        if not os.path.exists(traffic_model_path):
            raise FileNotFoundError(f"Traffic light model not found at {traffic_model_path}")
        self.traffic_light_model = load_trusted_model(traffic_model_path)
        
        # Lane detection model path
        self.lane_model_path = os.path.join("models", "lane_detection", "tusimple_18.pth")
        if not os.path.exists(self.lane_model_path):
            raise FileNotFoundError(f"Lane detection model not found at {self.lane_model_path}")
        
    def start(self):
        """Start video capture with fallback options"""
        if not self.is_running:
            # Try different camera indices
            for idx in [0, 1, 2]:  # Try first 3 camera indices
                print(f"Attempting to open camera {idx}")
                self.video = cv2.VideoCapture(idx)
                
                if self.video is None:
                    print(f"Failed to create VideoCapture object for index {idx}")
                    continue
                    
                if not self.video.isOpened():
                    print(f"Failed to open camera {idx}")
                    self.video.release()
                    continue
                    
                # Try reading a test frame
                ret, frame = self.video.read()
                if not ret or frame is None:
                    print(f"Failed to read test frame from camera {idx}")
                    self.video.release()
                    continue
                
                print(f"Successfully opened camera {idx}")
                self.camera_index = idx
                self.video.set(3, self.frameWidth)
                self.video.set(4, self.frameHeight)
                self.is_running = True
                return True
                
            # If we get here, no camera worked
            print("Failed to open any camera")
            return False
            
    def stop(self):
        """Stop video capture"""
        self.is_running = False
        if self.video is not None:
            self.video.release()
            self.video = None
            print("Camera released")
            
    def detect_lanes(self, frame):
        """Detect lanes in the frame"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blur, 50, 150)
            
            # Define region of interest
            height, width = edges.shape
            mask = np.zeros_like(edges)
            polygon = np.array([
                [(0, height), (width, height),
                 (width//2, height//2)]
            ])
            cv2.fillPoly(mask, polygon, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 15, 
                                  minLineLength=40, maxLineGap=20)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
            return True
        except Exception as e:
            print(f"Error in lane detection: {e}")
            return False
            
    def process_frame(self, frame):
        """Process a frame with both traffic light and lane detection"""
        try:
            # Resize frame to match our expected dimensions
            frame = cv2.resize(frame, (self.frameWidth, self.frameHeight))
            
            # Create a copy for drawing
            imgFinal = frame.copy()
            
            # Detect lanes
            lane_detected = self.detect_lanes(imgFinal)
            
            # Traffic light detection
            results = self.traffic_light_model(frame, conf=0.25)
            detections = {'red': 0, 'yellow': 0, 'green': 0}
            
            # Process traffic light detections
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    class_name = r.names[cls].lower()
                    if 'red' in class_name:
                        detections['red'] += 1
                        color = (0, 0, 255)
                    elif 'yellow' in class_name:
                        detections['yellow'] += 1
                        color = (0, 255, 255)
                    elif 'green' in class_name:
                        detections['green'] += 1
                        color = (0, 255, 0)
                    
                    cv2.rectangle(imgFinal, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(imgFinal, f'{class_name} {conf:.2f}',
                              (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add detection counts to the frame
            y_pos = 30
            for light, count in detections.items():
                if count > 0:
                    cv2.putText(imgFinal, f"{light}: {count}", 
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (255, 255, 255), 2)
                    y_pos += 30
            
            # Add lane detection status
            if lane_detected:
                cv2.putText(imgFinal, "Lanes Detected", 
                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 255, 0), 2)
            
            return imgFinal, detections
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame, {'red': 0, 'yellow': 0, 'green': 0}

detector = DriveSafeDetector()

def generate_frames(feed_type='processed'):
    while True:
        if not detector.is_running:
            print("Detector not running")
            break
            
        if detector.video is None:
            print("No video capture object")
            break
            
        success, frame = detector.video.read()
        if not success:
            print("Failed to read frame from camera")
            # Try to restart the camera
            detector.stop()
            if not detector.start():
                break
            continue
            
        if frame is None:
            print("Read frame is None")
            continue
            
        try:
            if feed_type == 'original':
                # Just resize the frame
                output_frame = cv2.resize(frame, (detector.frameWidth, detector.frameHeight))
            else:
                # Process frame for detection
                output_frame, _ = detector.process_frame(frame)
            
            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', output_frame)
            if not ret:
                print("Failed to encode frame")
                continue
                
            # Convert to bytes
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            continue

@app.route('/')
def index():
    # Ensure camera is released when page is loaded
    detector.stop()
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Get feed type from query parameters
    feed_type = request.args.get('feed', 'processed')
    
    # Start the detector if it's not running
    if not detector.is_running:
        detector.start()
    
    return Response(generate_frames(feed_type),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop')
def stop():
    detector.stop()
    return {"status": "stopped"}

if __name__ == '__main__':
    # Ensure camera is released when app starts
    detector.stop()
    try:
        app.run(debug=True, port=5001, host='0.0.0.0')
    finally:
        # Make sure to release the camera when the app exits
        detector.stop()
        if detector.video is not None:
            detector.video.release()
        cv2.destroyAllWindows() 