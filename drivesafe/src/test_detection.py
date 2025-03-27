import cv2
import numpy as np
from drivesafe.src.traffic_light_detector import TrafficLightDetector

def test_detection():
    # Initialize detector
    detector = TrafficLightDetector()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
        
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Detect traffic lights
        detections = detector.detect_traffic_lights(frame)
        
        # Draw detections
        frame = detector.draw_detections(frame, detections)
        
        # Display frame
        cv2.imshow('Traffic Light Detection Test', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_detection() 