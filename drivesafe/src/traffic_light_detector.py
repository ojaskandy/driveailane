from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Dict
import torch

class TrafficLightDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize the traffic light detector.
        
        Args:
            model_path (str, optional): Path to the YOLOv8 model weights. If None, uses default weights.
        """
        # Use the pre-trained YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Define traffic light class names
        self.traffic_light_classes = ['traffic light', 'stop light']
        
    def detect_traffic_lights(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect traffic lights in the given frame and determine their colors.
        
        Args:
            frame (np.ndarray): Input image frame
            
        Returns:
            List[Dict]: List of dictionaries containing detection information
        """
        results = self.model(frame)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class name and confidence
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names[class_id]
                confidence = box.conf[0].cpu().numpy()
                
                # Only process traffic light detections
                if class_name.lower() in self.traffic_light_classes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Extract the traffic light region
                    traffic_light_region = frame[int(y1):int(y2), int(x1):int(x2)]
                    
                    # Determine the color
                    color = self._determine_color(traffic_light_region)
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'color': color,
                        'class': class_name
                    })
        
        return detections
    
    def _determine_color(self, region: np.ndarray) -> str:
        """
        Determine the color of a traffic light region.
        
        Args:
            region (np.ndarray): Cropped region containing the traffic light
            
        Returns:
            str: Color of the traffic light ('red', 'yellow', 'green', or 'unknown')
        """
        if region.size == 0:
            return 'unknown'
            
        # Convert to HSV color space
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        
        green_lower = np.array([40, 100, 100])
        green_upper = np.array([80, 255, 255])
        
        # Create masks for each color
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Count pixels for each color
        red_pixels = np.sum(red_mask > 0)
        yellow_pixels = np.sum(yellow_mask > 0)
        green_pixels = np.sum(green_mask > 0)
        
        # Determine dominant color
        max_pixels = max(red_pixels, yellow_pixels, green_pixels)
        if max_pixels < 100:  # Threshold to avoid false positives
            return 'unknown'
            
        if max_pixels == red_pixels:
            return 'red'
        elif max_pixels == yellow_pixels:
            return 'yellow'
        else:
            return 'green'
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection boxes and labels on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            detections (List[Dict]): List of detection dictionaries
            
        Returns:
            np.ndarray: Frame with drawn detections
        """
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            color = detection['color']
            confidence = detection['confidence']
            class_name = detection['class']
            
            # Draw bounding box
            color_map = {
                'red': (0, 0, 255),
                'yellow': (0, 255, 255),
                'green': (0, 255, 0),
                'unknown': (128, 128, 128)
            }
            
            box_color = color_map.get(color, (128, 128, 128))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
            
            # Draw label
            label = f"{class_name} - {color} ({confidence:.2f})"
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        return frame 