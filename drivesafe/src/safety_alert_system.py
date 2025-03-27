from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

class AlertLevel(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

@dataclass
class Alert:
    level: AlertLevel
    message: str
    timestamp: float

class SafetyAlertSystem:
    def __init__(self, 
                 speed_threshold: float = 30.0,  # km/h
                 distance_threshold: float = 50.0,  # meters
                 time_to_intersection: float = 3.0):  # seconds
        """
        Initialize the safety alert system.
        
        Args:
            speed_threshold (float): Speed threshold in km/h for warning
            distance_threshold (float): Distance threshold in meters for warning
            time_to_intersection (float): Time to intersection threshold in seconds
        """
        self.speed_threshold = speed_threshold
        self.distance_threshold = distance_threshold
        self.time_to_intersection = time_to_intersection
        self.alerts: List[Alert] = []
        
    def analyze_detection(self, 
                         detection: Dict,
                         current_speed: float,
                         distance_to_light: float) -> Alert:
        """
        Analyze a traffic light detection and determine if an alert is needed.
        
        Args:
            detection (Dict): Traffic light detection information
            current_speed (float): Current vehicle speed in km/h
            distance_to_light (float): Distance to traffic light in meters
            
        Returns:
            Alert: Alert object with level and message
        """
        color = detection['color']
        confidence = detection['confidence']
        
        # Skip analysis if confidence is too low
        if confidence < 0.5:
            return Alert(AlertLevel.NONE, "", 0.0)
            
        # Calculate time to intersection
        time_to_intersection = (distance_to_light / 1000) / (current_speed / 3600)  # in seconds
        
        # Analyze based on traffic light color
        if color == 'red':
            if time_to_intersection < self.time_to_intersection:
                if current_speed > self.speed_threshold:
                    return Alert(
                        AlertLevel.HIGH,
                        "WARNING: Approaching red light at high speed!",
                        time_to_intersection
                    )
                else:
                    return Alert(
                        AlertLevel.MEDIUM,
                        "Caution: Red light ahead",
                        time_to_intersection
                    )
                    
        elif color == 'yellow':
            if time_to_intersection < self.time_to_intersection:
                if current_speed > self.speed_threshold:
                    return Alert(
                        AlertLevel.MEDIUM,
                        "Caution: Yellow light ahead, consider slowing down",
                        time_to_intersection
                    )
                else:
                    return Alert(
                        AlertLevel.LOW,
                        "Yellow light ahead",
                        time_to_intersection
                    )
                    
        elif color == 'green':
            if time_to_intersection < self.time_to_intersection:
                return Alert(
                    AlertLevel.LOW,
                    "Green light ahead",
                    time_to_intersection
                )
                
        return Alert(AlertLevel.NONE, "", 0.0)
    
    def update(self, 
               detections: List[Dict],
               current_speed: float,
               distance_to_light: float) -> List[Alert]:
        """
        Update the safety alert system with new detections and vehicle state.
        
        Args:
            detections (List[Dict]): List of traffic light detections
            current_speed (float): Current vehicle speed in km/h
            distance_to_light (float): Distance to nearest traffic light in meters
            
        Returns:
            List[Alert]: List of active alerts
        """
        self.alerts = []
        
        for detection in detections:
            alert = self.analyze_detection(detection, current_speed, distance_to_light)
            if alert.level != AlertLevel.NONE:
                self.alerts.append(alert)
                
        return self.alerts
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Get the list of currently active alerts.
        
        Returns:
            List[Alert]: List of active alerts
        """
        return self.alerts
    
    def get_highest_priority_alert(self) -> Alert:
        """
        Get the highest priority alert currently active.
        
        Returns:
            Alert: Highest priority alert, or NONE if no alerts are active
        """
        if not self.alerts:
            return Alert(AlertLevel.NONE, "", 0.0)
            
        return max(self.alerts, key=lambda x: x.level) 