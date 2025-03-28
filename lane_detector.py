import cv2
import numpy as np
from collections import deque

# Constants
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
CANNY_LOW = 50
CANNY_HIGH = 150
MIN_LINE_LENGTH = 100
MAX_LINE_GAP = 50
LANE_DEPARTURE_THRESHOLD = 50
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to keep a lane detection
MAX_LANE_WIDTH = 500  # Maximum reasonable lane width in pixels
MIN_LANE_WIDTH = 100  # Minimum reasonable lane width in pixels

class LaneDetector:
    def __init__(self):
        # History for smoothing
        self.left_lines_history = deque(maxlen=3)  # Reduced from 5 to 3 for faster updates
        self.right_lines_history = deque(maxlen=3)
        self.left_confidence = 0.0
        self.right_confidence = 0.0
        self.frame_count = 0

    def create_roi_mask(self, image):
        """Create a mask for the region of interest using a trapezoid shape"""
        mask = np.zeros_like(image)
        height, width = image.shape[:2]
        
        # Define a trapezoid ROI
        polygon = np.array([
            [(0, height),  # Bottom left
             (width, height),  # Bottom right
             (width//2 + 100, height//2),  # Top right
             (width//2 - 100, height//2)]  # Top left
        ], np.int32)
        
        cv2.fillPoly(mask, [polygon], 255)
        return mask

    def average_lines(self, lines, height):
        """Average multiple lines into a single lane line"""
        if not lines:
            return None

        x1_mean = np.mean([line[0][0] for line in lines])
        y1_mean = np.mean([line[0][1] for line in lines])
        x2_mean = np.mean([line[0][2] for line in lines])
        y2_mean = np.mean([line[0][3] for line in lines])

        if x2_mean - x1_mean == 0:  # Avoid division by zero
            return None

        slope = (y2_mean - y1_mean) / (x2_mean - x1_mean)
        intercept = y1_mean - slope * x1_mean

        # Calculate bottom and top points
        y_bottom = height
        x_bottom = int((y_bottom - intercept) / slope)
        y_top = height // 2
        x_top = int((y_top - intercept) / slope)

        return (x_bottom, y_bottom, x_top, y_top)

    def smooth_lane(self, current_lane, prev_lanes):
        """Apply temporal smoothing to lane detection"""
        if current_lane is None:
            return prev_lanes[-1] if prev_lanes else None
            
        prev_lanes.append(current_lane)
        smoothed = tuple(map(lambda x: int(sum(x) / len(x)), zip(*prev_lanes)))
        return smoothed

    def validate_lanes(self, left_lane, right_lane, frame_width):
        """Validate detected lanes based on various criteria"""
        if left_lane is None or right_lane is None:
            return False

        # Check lane width
        lane_width = abs(left_lane[0] - right_lane[0])  # Width at bottom
        if not (MIN_LANE_WIDTH < lane_width < MAX_LANE_WIDTH):
            return False

        # Check if lanes cross
        left_slope = (left_lane[3] - left_lane[1]) / (left_lane[2] - left_lane[0]) if left_lane[2] != left_lane[0] else float('inf')
        right_slope = (right_lane[3] - right_lane[1]) / (right_lane[2] - right_lane[0]) if right_lane[2] != right_lane[0] else float('inf')
        if left_slope * right_slope > 0:  # Slopes should have opposite signs
            return False

        # Check if lanes are within frame
        if (min(left_lane[0], left_lane[2]) < 0 or 
            max(right_lane[0], right_lane[2]) > frame_width):
            return False

        return True

    def calculate_confidence(self, lines, frame_width):
        """Calculate confidence score for detected lines"""
        if not lines:
            return 0.0

        # Count number of supporting line segments
        num_segments = len(lines)
        
        # Calculate average length of line segments
        avg_length = np.mean([
            np.sqrt((line[0][2] - line[0][0])**2 + (line[0][3] - line[0][1])**2)
            for line in lines
        ])
        
        # Normalize length score (0 to 1)
        length_score = min(avg_length / MIN_LINE_LENGTH, 1.0)
        
        # Calculate position score (higher if lines are where we expect them)
        positions = [(line[0][0] + line[0][2])/2 for line in lines]
        avg_position = np.mean(positions)
        position_score = 1.0
        if avg_position < frame_width/2:  # Left lane
            position_score = 1.0 - (avg_position / (frame_width/2))
        else:  # Right lane
            position_score = (avg_position - frame_width/2) / (frame_width/2)

        # Combine scores
        confidence = (0.4 * min(num_segments/5, 1.0) +  # Number of segments
                     0.4 * length_score +                # Length of segments
                     0.2 * position_score)              # Position score

        return confidence

    def update_lane_confidence(self, left_lines, right_lines, frame_width):
        """Update confidence scores for left and right lanes"""
        # Calculate new confidence scores
        new_left_conf = self.calculate_confidence(left_lines, frame_width)
        new_right_conf = self.calculate_confidence(right_lines, frame_width)
        
        # Update smoothed confidence scores
        self.left_confidence = 0.7 * self.left_confidence + 0.3 * new_left_conf
        self.right_confidence = 0.7 * self.right_confidence + 0.3 * new_right_conf

    def process_frame(self, frame):
        """Process a frame to detect and draw lanes"""
        self.frame_count += 1
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
        
        # Create and apply ROI mask
        roi_mask = self.create_roi_mask(edges)
        masked_edges = cv2.bitwise_and(edges, roi_mask)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            masked_edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50,
            minLineLength=MIN_LINE_LENGTH,
            maxLineGap=MAX_LINE_GAP
        )
        
        # Separate left and right lanes
        left_lines, right_lines = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:  # Skip vertical lines
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                if slope < -0.5:  # Left lane
                    left_lines.append(line)
                elif slope > 0.5:  # Right lane
                    right_lines.append(line)
        
        # Get averaged lanes
        height = frame.shape[0]
        left_lane = self.average_lines(left_lines, height)
        right_lane = self.average_lines(right_lines, height)
        
        # Update history for temporal smoothing
        if left_lines:
            self.left_lines_history.append(left_lane)
        if right_lines:
            self.right_lines_history.append(right_lane)
        
        # Use most recent valid detection if current detection fails
        if not left_lane and self.left_lines_history:
            left_lane = self.left_lines_history[-1]
        if not right_lane and self.right_lines_history:
            right_lane = self.right_lines_history[-1]
        
        # Draw the lanes and warnings
        result = self.draw_lanes(frame, left_lane, right_lane)
        
        return result

    def draw_lanes(self, frame, left_lane, right_lane):
        """Draw detected lanes and warnings on the frame"""
        line_image = np.zeros_like(frame)
        
        # Draw confidence scores
        cv2.putText(frame, 
                    f"L Conf: {self.left_confidence:.2f}", 
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0) if self.left_confidence > CONFIDENCE_THRESHOLD else (0, 0, 255),
                    2)
        cv2.putText(frame, 
                    f"R Conf: {self.right_confidence:.2f}", 
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0) if self.right_confidence > CONFIDENCE_THRESHOLD else (0, 0, 255),
                    2)
        
        if left_lane:
            cv2.line(line_image, 
                     (left_lane[0], left_lane[1]), 
                     (left_lane[2], left_lane[3]), 
                     (0, 255, 0), 5)  # Green for left lane
            
        if right_lane:
            cv2.line(line_image, 
                     (right_lane[0], right_lane[1]), 
                     (right_lane[2], right_lane[3]), 
                     (0, 0, 255), 5)  # Red for right lane
        
        # Blend the lines with the original frame
        result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
        
        # Add lane departure warning
        if left_lane and right_lane:
            lane_center = (left_lane[0] + right_lane[0]) // 2
            vehicle_center = frame.shape[1] // 2
            offset = vehicle_center - lane_center
            
            if abs(offset) > LANE_DEPARTURE_THRESHOLD:
                cv2.putText(result, 
                           "WARNING: Lane Departure!", 
                           (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, 
                           (0, 0, 255), 
                           2)
        
        # Add status overlay
        cv2.putText(result, 
                    "Lane Detection Active", 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2)
        
        return result 