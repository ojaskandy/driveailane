import cv2
import numpy as np
from ultralytics import YOLO
import os
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
    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load
    
    try:
        model = YOLO(model_path)
        return model
    finally:
        torch.load = original_torch_load

class DriveSafeGUI:
    def __init__(self):
        # Frame dimensions
        self.frameWidth = 640
        self.frameHeight = 480
        
        # Window names
        self.WINDOW_ORIGINAL = "Original Feed"
        self.WINDOW_PROCESSED = "Processed Feed (Traffic Lights & Lanes)"
        
        self.setup_models()
        self.setup_camera()
        self.is_running = False
        
    def setup_models(self):
        """Load detection models"""
        # Traffic light model path
        traffic_model_path = os.path.join("models", "traffic_light", "best_traffic_small_yolo.pt")
        if not os.path.exists(traffic_model_path):
            raise FileNotFoundError(f"Traffic light model not found at {traffic_model_path}")
        print("Loading traffic light model...")
        self.traffic_light_model = load_trusted_model(traffic_model_path)
        print("Traffic light model loaded successfully")
        
    def setup_camera(self):
        """Initialize the camera"""
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
            
        self.cap.set(3, self.frameWidth)
        self.cap.set(4, self.frameHeight)
        print("Camera initialized successfully")
        
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
            # Create a copy for drawing
            processed_frame = frame.copy()
            
            # Detect lanes
            lane_detected = self.detect_lanes(processed_frame)
            
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
                    
                    cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(processed_frame, f'{class_name} {conf:.2f}',
                              (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add detection counts to the frame
            y_pos = 30
            for light, count in detections.items():
                if count > 0:
                    cv2.putText(processed_frame, f"{light}: {count}", 
                              (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (255, 255, 255), 2)
                    y_pos += 30
            
            # Add lane detection status
            if lane_detected:
                cv2.putText(processed_frame, "Lanes Detected", 
                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.7, (0, 255, 0), 2)
            
            return processed_frame
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame
            
    def run(self):
        """Main loop for the GUI application"""
        print("Starting GUI application...")
        print("Press 'q' to quit")
        
        cv2.namedWindow(self.WINDOW_ORIGINAL)
        cv2.namedWindow(self.WINDOW_PROCESSED)
        
        # Position windows side by side
        cv2.moveWindow(self.WINDOW_ORIGINAL, 0, 0)
        cv2.moveWindow(self.WINDOW_PROCESSED, self.frameWidth + 30, 0)
        
        self.is_running = True
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
                
            # Resize frame
            frame = cv2.resize(frame, (self.frameWidth, self.frameHeight))
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Show frames
            cv2.imshow(self.WINDOW_ORIGINAL, frame)
            cv2.imshow(self.WINDOW_PROCESSED, processed_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed")

if __name__ == "__main__":
    try:
        app = DriveSafeGUI()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        cv2.destroyAllWindows() 