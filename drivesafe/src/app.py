import cv2
import numpy as np
from typing import Optional, Tuple
import time
from .traffic_light_detector import TrafficLightDetector
from .safety_alert_system import SafetyAlertSystem, Alert, AlertLevel

class DriveSafe:
    def __init__(self,
                 camera_id: int = 0,
                 speed_threshold: float = 30.0,
                 distance_threshold: float = 50.0,
                 time_to_intersection: float = 3.0):
        """
        Initialize the DriveSafe application.
        
        Args:
            camera_id (int): ID of the camera to use
            speed_threshold (float): Speed threshold for warnings in km/h
            distance_threshold (float): Distance threshold for warnings in meters
            time_to_intersection (float): Time to intersection threshold in seconds
        """
        self.detector = TrafficLightDetector()
        self.alert_system = SafetyAlertSystem(
            speed_threshold=speed_threshold,
            distance_threshold=distance_threshold,
            time_to_intersection=time_to_intersection
        )
        self.camera_id = camera_id
        self.camera = None
        self.is_running = False
        self.current_speed = 0.0  # km/h
        self.distance_to_light = 0.0  # meters
        
    def start(self):
        """Start the DriveSafe application."""
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            raise RuntimeError("Failed to open camera")
            
        self.is_running = True
        self._main_loop()
        
    def stop(self):
        """Stop the DriveSafe application."""
        self.is_running = False
        if self.camera:
            self.camera.release()
            
    def set_vehicle_state(self, speed: float, distance_to_light: float):
        """
        Update the vehicle's current state.
        
        Args:
            speed (float): Current speed in km/h
            distance_to_light (float): Distance to nearest traffic light in meters
        """
        self.current_speed = speed
        self.distance_to_light = distance_to_light
        
    def _main_loop(self):
        """Main application loop."""
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Detect traffic lights
            detections = self.detector.detect_traffic_lights(frame)
            
            # Update alert system
            alerts = self.alert_system.update(
                detections,
                self.current_speed,
                self.distance_to_light
            )
            
            # Draw detections and alerts
            frame = self.detector.draw_detections(frame, detections)
            frame = self._draw_alerts(frame, alerts)
            
            # Display the frame
            cv2.imshow('DriveSafe', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cv2.destroyAllWindows()
        
    def _draw_alerts(self, frame: np.ndarray, alerts: list[Alert]) -> np.ndarray:
        """
        Draw alerts on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            alerts (list[Alert]): List of active alerts
            
        Returns:
            np.ndarray: Frame with drawn alerts
        """
        if not alerts:
            return frame
            
        # Get highest priority alert
        alert = max(alerts, key=lambda x: x.level)
        
        # Define alert colors
        color_map = {
            AlertLevel.HIGH: (0, 0, 255),
            AlertLevel.MEDIUM: (0, 165, 255),
            AlertLevel.LOW: (0, 255, 0)
        }
        
        # Draw alert box
        height, width = frame.shape[:2]
        box_height = 60
        box_y = height - box_height - 10
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (10, box_y), 
                     (width - 10, height - 10),
                     (0, 0, 0), 
                     -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw alert text
        color = color_map.get(alert.level, (255, 255, 255))
        cv2.putText(frame, 
                   alert.message,
                   (20, box_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.7,
                   color,
                   2)
        
        return frame 