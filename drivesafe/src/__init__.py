"""
DriveSafe - Traffic Light Detection and Safety Alert System
"""

from .traffic_light_detector import TrafficLightDetector
from .safety_alert_system import SafetyAlertSystem, Alert, AlertLevel
from .app import DriveSafe

__version__ = '0.1.0'
__all__ = ['TrafficLightDetector', 'SafetyAlertSystem', 'Alert', 'AlertLevel', 'DriveSafe'] 