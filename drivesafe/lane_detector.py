import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        # Parameters for lane detection
        self.canny_low = 50
        self.canny_high = 150
        self.rho = 1
        self.theta = np.pi/180
        self.min_threshold = 30
        self.min_line_length = 40
        self.max_line_gap = 20
        
        # Line smoothing
        self.left_lines = []
        self.right_lines = []
        self.smooth_factor = 5
        
        # Region of interest parameters
        self.roi_vertices = None
    
    def detect_edges(self, frame):
        """Apply Canny edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        return edges
    
    def region_of_interest(self, edges):
        """Apply region of interest mask"""
        if self.roi_vertices is None:
            height, width = edges.shape
            # Define a trapezoid shape for our region of interest
            self.roi_vertices = np.array([
                [(0, height),
                 (width * 0.45, height * 0.6),  # Top left
                 (width * 0.55, height * 0.6),  # Top right
                 (width, height)]
            ], dtype=np.int32)
        
        # Create mask
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self.roi_vertices, 255)
        
        # Apply mask
        masked_edges = cv2.bitwise_and(edges, mask)
        return masked_edges
    
    def detect_line_segments(self, edges):
        """Detect line segments using Hough transform"""
        lines = cv2.HoughLinesP(
            edges,
            rho=self.rho,
            theta=self.theta,
            threshold=self.min_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        return lines if lines is not None else []
    
    def average_slope_intercept(self, lines):
        """Calculate average slope and intercept for left and right lanes"""
        left_lines = []
        right_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:  # Skip vertical lines
                continue
            
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            
            # Filter based on slope
            if abs(slope) < 0.1:  # Skip horizontal lines
                continue
                
            if slope < 0:  # Left lane
                left_lines.append((slope, intercept))
            else:  # Right lane
                right_lines.append((slope, intercept))
        
        # Update line history for smoothing
        if left_lines:
            left_avg = np.mean(left_lines, axis=0)
            self.left_lines.append(left_avg)
            if len(self.left_lines) > self.smooth_factor:
                self.left_lines.pop(0)
        
        if right_lines:
            right_avg = np.mean(right_lines, axis=0)
            self.right_lines.append(right_avg)
            if len(self.right_lines) > self.smooth_factor:
                self.right_lines.pop(0)
        
        # Get smoothed averages
        left_avg = np.mean(self.left_lines, axis=0) if self.left_lines else None
        right_avg = np.mean(self.right_lines, axis=0) if self.right_lines else None
        
        return left_avg, right_avg
    
    def draw_lane_lines(self, frame, left_line, right_line):
        """Draw lane lines on the frame"""
        line_image = np.zeros_like(frame)
        
        if left_line is not None:
            slope, intercept = left_line
            y1 = frame.shape[0]
            y2 = int(frame.shape[0] * 0.6)
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        
        if right_line is not None:
            slope, intercept = right_line
            y1 = frame.shape[0]
            y2 = int(frame.shape[0] * 0.6)
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Blend the line image with the original frame
        return cv2.addWeighted(frame, 0.8, line_image, 1.0, 0.0)
    
    def process_frame(self, frame):
        """Process a frame to detect and draw lanes"""
        try:
            # Detect edges
            edges = self.detect_edges(frame)
            
            # Apply region of interest
            roi_edges = self.region_of_interest(edges)
            
            # Detect line segments
            line_segments = self.detect_line_segments(roi_edges)
            
            if len(line_segments) > 0:
                # Calculate average lines
                left_line, right_line = self.average_slope_intercept(line_segments)
                
                # Draw lane lines
                frame = self.draw_lane_lines(frame, left_line, right_line)
            
            return frame
            
        except Exception as e:
            print(f"Error in lane detection: {e}")
            return frame 