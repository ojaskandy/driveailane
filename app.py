from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os
import smtplib
from email.mime.text import MIMEText
from threading import Lock
import queue
from twilio.rest import Client
from datetime import datetime, timedelta
import json
import threading
from queue import Queue
import time
import pygame
import math
import collections
from scipy.optimize import lsq_linear
from collections import deque

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
thread_lock = Lock()

# Twilio credentials
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')
TARGET_PHONE_NUMBER = '+14256230692'
TARGET_EMAIL = 'ojaskandy@gmail.com'

# Queue for storing email addresses
email_queue = queue.Queue()

# Store active trips
active_trips = {}
frame_queues = {}

# Trip tracking
class TripStats:
    def __init__(self):
        self.total_drives = 0
        self.total_time = 0  # in seconds
        self.last_drive = None
        self.current_trips = {}  # store active trips

    def format_location(self, location):
        if isinstance(location, dict):
            return f"{location['lat']:.6f}, {location['lng']:.6f}"
        return str(location)

trip_stats = TripStats()

# Add these as global variables after other globals
line_history = collections.deque(maxlen=5)  # Store last 5 frames of line positions
dashboard_height_ratio = 0.3  # 30% of frame height for dashboard

# Constants from their implementation
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
LANE_WIDTH = 1.5
BOUNDARY_THRESH = 1.8
HORIZON = -40.1
FILTER_STRENGTH = 1

class LaneDetector:
    def __init__(self):
        self.v = np.arange(0, IMAGE_HEIGHT, 1)
        self.u = np.arange(0, IMAGE_WIDTH, 1)
        self.threshold = BOUNDARY_THRESH
        self.lane_width = LANE_WIDTH
        self.h = HORIZON
        self.k = 0
        self.bl = 0
        self.br = 0
        self.bc = 0
        self.c = 0
        self.bcdq = deque(maxlen=FILTER_STRENGTH)
        self.bldq = deque(maxlen=FILTER_STRENGTH)
        self.brdq = deque(maxlen=FILTER_STRENGTH)
        self.left_lane_points = np.array([])
        self.right_lane_points = np.array([])
        self.lane = np.array([])
        self.low_b = np.array([-500000, -8, -8, IMAGE_WIDTH/2 -20])
        self.up_b = np.array([500000, 8, 8, IMAGE_WIDTH/2 +20])

    def hyperbola_pair(self, b):
        return self.k/(self.v-self.h)+b*(self.v-self.h)+self.c

    def process_frame(self, frame):
        # Resize frame
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Sobel edge detection
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        
        # Apply binary threshold
        _, binary = cv2.threshold(scaled_sobel, 50, 255, cv2.THRESH_BINARY)
        
        # Get lane points
        self.get_initial_lane_points(binary)
        
        # Solve for lane parameters
        self.solve_lane()
        
        # Draw detected lanes
        result = self.draw_lanes(frame)
        
        return result

    def get_initial_lane_points(self, edge_image):
        image_height = edge_image.shape[0]
        image_width = edge_image.shape[1]

        left_lane_points = np.empty((image_height, 1))
        left_lane_points[:] = np.NAN
        right_lane_points = np.empty((image_height, 1))
        right_lane_points[:] = np.NAN

        lane_numbers = np.arange(image_width)
        edge_image = edge_image / 255

        for row in range(image_height-1, -1, -1):
            curr_row = np.multiply((lane_numbers - image_height), edge_image[row, :])
            points_to_the_right = np.where(curr_row > 0)[0]
            points_to_the_left = np.where(curr_row < 0)[0]
            if points_to_the_right.size > 0:
                right_lane_points[row] = np.amin(points_to_the_right)
            if points_to_the_left.size > 0:
                left_lane_points[row] = np.amax(points_to_the_left)
            if row == 300:
                break
        self.left_lane_points = left_lane_points
        self.right_lane_points = right_lane_points

    def solve_lane(self):
        A, b = self.preprocess_for_solving()
        self.solving_lane_params(A, b)

    def preprocess_for_solving(self):
        l = self.left_lane_points
        r = self.right_lane_points
        l_ind = ~np.isnan(l)
        r_ind = ~np.isnan(r)
        l_num = l[l_ind]
        r_num = r[r_ind]
        vl = self.v[l_ind.flatten()]
        vr = self.v[r_ind.flatten()]
        l_num = l_num.reshape((len(l_num), 1))
        r_num = r_num.reshape((len(r_num), 1))
        vl = vl.reshape(l_num.shape)
        vr = vr.reshape(r_num.shape)

        lh = (vl-self.h)
        lA = 1/lh
        rh = (vr-self.h)
        rA = 1/rh
        ones = np.ones(l_num.shape)
        zeros = np.zeros(l_num.shape)
        LA = np.hstack((np.hstack((lA, lh)), np.hstack((zeros, ones))))
        ones = np.ones(r_num.shape)
        zeros = np.zeros(r_num.shape)
        RA = np.hstack((np.hstack((rA, zeros)), np.hstack((rh, ones))))
        A = np.vstack((LA, RA))
        b = (np.concatenate((l_num, r_num))).flatten()
        return A, b

    def solving_lane_params(self, A, b):
        x = lsq_linear(A, b, bounds=(self.low_b, self.up_b), method='bvls', max_iter=3).x
        self.k = x[0]
        self.bl = x[1]
        self.br = x[2]
        self.c = x[3]
        self.bc = (x[1]+x[2])/2
        self.lane = self.hyperbola_pair(self.bc)

    def draw_lanes(self, frame):
        if len(self.lane) > 0:
            # Draw left lane
            left_points = np.column_stack((self.hyperbola_pair(self.bl).astype(np.int32), self.v.astype(np.int32)))
            left_points = left_points[left_points[:, 0] >= 0]
            left_points = left_points[left_points[:, 0] < frame.shape[1]]
            cv2.polylines(frame, [left_points], False, (0, 0, 255), 2)

            # Draw right lane
            right_points = np.column_stack((self.hyperbola_pair(self.br).astype(np.int32), self.v.astype(np.int32)))
            right_points = right_points[right_points[:, 0] >= 0]
            right_points = right_points[right_points[:, 0] < frame.shape[1]]
            cv2.polylines(frame, [right_points], False, (0, 0, 255), 2)

            # Draw center lane
            center_points = np.column_stack((self.lane.astype(np.int32), self.v.astype(np.int32)))
            center_points = center_points[center_points[:, 0] >= 0]
            center_points = center_points[center_points[:, 0] < frame.shape[1]]
            cv2.polylines(frame, [center_points], False, (0, 255, 0), 2)

        return frame

lane_detector = LaneDetector()

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=3)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

# Cache for region of interest vertices
roi_cache = {}

def get_roi_vertices(height, width):
    cache_key = f"{height}_{width}"
    if cache_key not in roi_cache:
        roi_cache[cache_key] = np.array([[
            (0, height),
            (width//2, height//2),
            (width, height)
        ]], dtype=np.int32)
    return roi_cache[cache_key]

def extrapolate_line(lines, height):
    if not lines:
        return None
    slopes = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in [line[0]] if x2 - x1 != 0]
    if not slopes:
        return None
    avg_slope = np.mean(slopes)
    avg_x = np.mean([x1 for line in lines for x1, _, _, _ in [line[0]]])
    avg_y = np.mean([y1 for line in lines for _, y1, _, _ in [line[0]]])
    
    y_bottom = height
    x_bottom = int(avg_x + (y_bottom - avg_y) / avg_slope) if avg_slope != 0 else int(avg_x)
    y_top = int(height * 0.4)  # Extend to 40% of frame height
    x_top = int(avg_x + (y_top - avg_y) / avg_slope) if avg_slope != 0 else int(avg_x)
    
    return np.array([[x_top, y_top, x_bottom, y_bottom]])

def smooth_lines(new_lines, side='left'):
    global line_history
    if new_lines is None:
        return None
    
    line_history.append(new_lines)
    if len(line_history) < 2:
        return new_lines
        
    # Calculate weighted average of line positions
    weights = np.linspace(0.5, 1.0, len(line_history))
    weighted_lines = np.average(line_history, axis=0, weights=weights)
    return weighted_lines.astype(int)

def detect_dashboard_edge(frame, height, width):
    # Look at bottom half of frame
    bottom_half = frame[height//2:, :]
    edges = cv2.Canny(bottom_half, 50, 150)
    
    # Find horizontal lines
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=width//2,
        maxLineGap=50
    )
    
    if lines is not None:
        # Find highest horizontal line (lowest y-value)
        dashboard_y = min(line[0][1] for line in lines) + height//2
        return min(dashboard_y, int(height * 0.4))  # Cap at 40% of frame height
    
    return int(height * dashboard_height_ratio)

def process(image):
    try:
        height, width = image.shape[:2]
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Edge detection with optimized parameters
        canny_image = cv2.Canny(blurred, 50, 150)
        
        # Get cached ROI vertices
        vertices = get_roi_vertices(height, width)
        
        # Apply region of interest mask
        cropped_image = region_of_interest(canny_image, vertices)
        
        # Detect lines with optimized parameters
        lines = cv2.HoughLinesP(
            cropped_image,
            rho=1,
            theta=np.pi/180,
            threshold=20,
            minLineLength=20,
            maxLineGap=300
        )
        
        # Draw lines
        result = draw_the_lines(image, lines)
        return result
    except Exception as e:
        print(f"Error in process function: {str(e)}")
        return image

def encode_frame(frame):
    try:
        # Compress image with lower quality for faster transmission
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error encoding frame: {str(e)}")
        return None

def notify_new_signup(email):
    try:
        # Send email
        msg = MIMEText(f"New waitlist signup: {email}")
        msg['Subject'] = 'DriveAI Waitlist Signup'
        msg['From'] = 'noreply@driveai.com'
        msg['To'] = TARGET_EMAIL
        
        # Print email for now (replace with actual email sending later)
        print(f"Would send email to {TARGET_EMAIL}: {email}")
        
        # Send SMS via Twilio
        if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            message = client.messages.create(
                body=f"New DriveAI waitlist signup: {email}",
                from_=TWILIO_PHONE_NUMBER,
                to=TARGET_PHONE_NUMBER
            )
            print(f"SMS sent: {message.sid}")
        
        return True
    except Exception as e:
        print(f"Error in notification: {str(e)}")
        return False

def process_email_queue():
    """Process emails immediately and send notifications"""
    while True:
        try:
            while not email_queue.empty():
                email = email_queue.get_nowait()
                notify_new_signup(email)
            
            socketio.sleep(5)  # Check queue every 5 seconds
            
        except Exception as e:
            print(f"Error processing email queue: {str(e)}")
            socketio.sleep(60)  # Wait a minute before retrying on error

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/drive')
def drive():
    return render_template('drive.html')

@app.route('/join-waitlist', methods=['POST'])
def join_waitlist():
    data = request.get_json()
    if data and 'email' in data:
        email = data.get('email')
        if notify_new_signup(email):
            return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Invalid email'}), 400

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    emit('stats_update', {
        'total_drives': trip_stats.total_drives,
        'total_time': format_duration(trip_stats.total_time),
        'last_drive': trip_stats.last_drive or 'Never'
    })

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    # Clean up any active trips for this client
    if request.sid in active_trips:
        del active_trips[request.sid]
    if request.sid in frame_queues:
        del frame_queues[request.sid]

@socketio.on('frame')
def handle_frame(data):
    if data.startswith('data:image'):
        # Extract the actual base64 data
        base64_data = data.split(',')[1]
    else:
        base64_data = data

    # Decode base64 image
    image_data = base64.b64decode(base64_data)
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is not None:
        # Process frame using lane detection
        processed_frame = lane_detector.process_frame(frame)
        
        # Encode processed frame to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send processed frame back to client
        socketio.emit('processed_frame', processed_base64)

@socketio.on('start_trip')
def handle_start_trip(data):
    trip_id = request.sid
    start_location = data.get('location', 'Location unavailable')
    start_time = datetime.now()
    
    active_trips[trip_id] = {
        'start_time': start_time,
        'start_location': start_location,
        'frames_processed': 0
    }
    
    # Create a queue for this trip's frames
    frame_queues[trip_id] = Queue()
    
    # Start a worker thread for this trip
    worker_thread = threading.Thread(target=process_frames_worker, args=(trip_id,))
    worker_thread.daemon = True
    worker_thread.start()
    
    return {'trip_id': trip_id}

@socketio.on('end_trip')
def handle_end_trip(data):
    trip_id = request.sid
    if trip_id in active_trips:
        trip_data = active_trips[trip_id]
        end_time = datetime.now()
        end_location = data.get('location', 'Location unavailable')
        
        duration = end_time - trip_data['start_time']
        duration_str = str(duration).split('.')[0]  # Format as HH:MM:SS
        
        # Clean up
        del active_trips[trip_id]
        if trip_id in frame_queues:
            del frame_queues[trip_id]
        
        # Calculate trip statistics
        trip_stats.total_drives += 1
        trip_stats.total_time += duration.total_seconds()
        trip_stats.last_drive = end_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format locations for display
        start_loc = trip_stats.format_location(trip_data['start_location'])
        end_loc = trip_stats.format_location(end_location)
        
        # Prepare trip summary
        trip_summary = {
            'duration': duration_str,
            'start_location': start_loc,
            'end_location': end_loc,
            'start_time': trip_data['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'frames_processed': trip_data['frames_processed']
        }
        
        # Emit updated stats to all clients
        socketio.emit('stats_update', {
            'total_drives': trip_stats.total_drives,
            'total_time': format_duration(trip_stats.total_time),
            'last_drive': trip_stats.last_drive
        })
        
        return trip_summary
    return {'error': 'Trip not found'}

def format_duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"

def process_frames_worker(trip_id):
    while trip_id in active_trips:
        try:
            if trip_id in frame_queues and not frame_queues[trip_id].empty():
                frame_data = frame_queues[trip_id].get()
                processed_frame = process_frame(frame_data)
                if processed_frame:
                    socketio.emit('processed_frame', processed_frame, room=trip_id)
            else:
                time.sleep(0.01)  # Short sleep to prevent CPU overuse
        except Exception as e:
            print(f"Error in worker thread: {str(e)}")
            continue

# Initialize pygame mixer for audio
pygame.mixer.init()

# Constants for lane departure detection
LANE_DEPARTURE_THRESHOLD = 0.15  # 15% of frame width
MIN_BEEP_INTERVAL = 2  # Minimum seconds between beeps
last_beep_time = datetime.now() - timedelta(seconds=MIN_BEEP_INTERVAL)

def calculate_lane_position(lines, frame_width):
    if lines is None:
        return None, None

    left_lines = []
    right_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:  # Skip vertical lines
            continue
            
        slope = (y2 - y1) / (x2 - x1)
        
        # Filter lines by slope
        if 0.3 < abs(slope) < 2:  # Reasonable slope range for lane lines
            if slope < 0:  # Left lane
                left_lines.append(line)
            else:  # Right lane
                right_lines.append(line)
    
    # Calculate average positions
    left_pos = None
    right_pos = None
    
    if left_lines:
        left_x = sum(line[0][0] for line in left_lines) / len(left_lines)
        left_pos = left_x / frame_width
        
    if right_lines:
        right_x = sum(line[0][0] for line in right_lines) / len(right_lines)
        right_pos = right_x / frame_width
        
    return left_pos, right_pos

def check_lane_departure(left_pos, right_pos, frame_width):
    global last_beep_time
    
    if left_pos is None and right_pos is None:
        return None
        
    # Calculate car position relative to detected lanes
    if left_pos is not None and right_pos is not None:
        car_pos = 0.5  # Assume car is in center of frame
        lane_center = (left_pos + right_pos) / 2
        offset = car_pos - lane_center
        
        # Check if car is too close to either lane
        current_time = datetime.now()
        if (abs(offset) > LANE_DEPARTURE_THRESHOLD and 
            (current_time - last_beep_time).total_seconds() >= MIN_BEEP_INTERVAL):
            last_beep_time = current_time
            return "left" if offset > 0 else "right"
            
    # If only one lane is detected, check distance to that lane
    elif left_pos is not None:
        if left_pos > LANE_DEPARTURE_THRESHOLD:
            current_time = datetime.now()
            if (current_time - last_beep_time).total_seconds() >= MIN_BEEP_INTERVAL:
                last_beep_time = current_time
                return "left"
                
    elif right_pos is not None:
        if right_pos < (1 - LANE_DEPARTURE_THRESHOLD):
            current_time = datetime.now()
            if (current_time - last_beep_time).total_seconds() >= MIN_BEEP_INTERVAL:
                last_beep_time = current_time
                return "right"
                
    return None

def process_frame(frame):
    try:
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Detect dashboard edge (or use static height)
        dashboard_y = detect_dashboard_edge(frame, height, width)
        
        # Create a copy for preprocessing
        working_frame = frame.copy()
        
        # Apply histogram equalization to improve contrast
        gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        
        # Convert to HSV and LAB color spaces
        hsv = cv2.cvtColor(working_frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(working_frame, cv2.COLOR_BGR2LAB)
        
        # Define color ranges for white and yellow lanes
        white_lower = np.array([0, 0, 200])    # Tightened for pure whites
        white_upper = np.array([180, 30, 255])
        yellow_lower = np.array([20, 100, 100]) # Narrowed hue range
        yellow_upper = np.array([30, 255, 255])
        
        # Create masks for white and yellow lanes in HSV
        white_mask_hsv = cv2.inRange(hsv, white_lower, white_upper)
        yellow_mask_hsv = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Create yellow mask in LAB space
        yellow_mask_lab = cv2.inRange(lab, np.array([50, 110, 110]), np.array([255, 145, 145]))
        
        # Combine masks
        yellow_mask = cv2.bitwise_or(yellow_mask_hsv, yellow_mask_lab)
        combined_mask = cv2.bitwise_or(white_mask_hsv, yellow_mask)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Dilate to enhance lane markings
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        
        # Create and apply dashboard mask
        dashboard_mask = np.ones_like(combined_mask)
        dashboard_mask[dashboard_y:, :] = 0
        combined_mask = cv2.bitwise_and(combined_mask, dashboard_mask)
        
        # Apply adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            equalized, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Combine with color mask
        edges = cv2.bitwise_and(adaptive_thresh, combined_mask)
        
        # Define ROI vertices
        roi_vertices = np.array([[
            (0, height),
            (0, dashboard_y),
            (width, dashboard_y),
            (width, height)
        ]], dtype=np.int32)
        
        # Apply ROI mask
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Apply Hough transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=20,
            minLineLength=50,
            maxLineGap=200
        )
        
        # Process detected lines
        line_image = np.zeros_like(frame)
        if lines is not None:
            left_lines = []
            right_lines = []
            
            # Previous frame's slopes for consistency check
            prev_left_slope = None
            prev_right_slope = None
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Skip if line is in dashboard area
                if y1 > dashboard_y or y2 > dashboard_y:
                    continue
                
                if x2 - x1 == 0:  # Skip vertical lines
                    continue
                
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter lines by slope and position
                if 0.2 < abs(slope) < 2.5:
                    # Check slope consistency if we have previous slopes
                    if slope < 0 and x1 < width/2:  # Left lane
                        if prev_left_slope is None or abs(slope - prev_left_slope) < 0.5:
                            left_lines.append(line)
                            prev_left_slope = slope
                    elif slope > 0 and x1 > width/2:  # Right lane
                        if prev_right_slope is None or abs(slope - prev_right_slope) < 0.5:
                            right_lines.append(line)
                            prev_right_slope = slope
            
            # Extrapolate and smooth lines
            left_line = extrapolate_line(left_lines, height)
            right_line = extrapolate_line(right_lines, height)
            
            # Apply temporal smoothing
            left_line = smooth_lines(left_line, 'left')
            right_line = smooth_lines(right_line, 'right')
            
            # Draw the smoothed lines
            if left_line is not None:
                cv2.line(line_image,
                        (left_line[0][0], left_line[0][1]),
                        (left_line[0][2], left_line[0][3]),
                        (0, 255, 0), 3)
            if right_line is not None:
                cv2.line(line_image,
                        (right_line[0][0], right_line[0][1]),
                        (right_line[0][2], right_line[0][3]),
                        (0, 255, 0), 3)
        
        # Combine original frame with line image
        result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
        
        # Draw dashboard area
        cv2.rectangle(result, (0, dashboard_y), (width, height), (0, 255, 255), 2)
        
        # Add text overlay
        cv2.putText(result, "Lane Detection Active", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Check for lane departure
        left_pos, right_pos = calculate_lane_position(lines, width)
        departure_direction = check_lane_departure(left_pos, right_pos, width)
        
        # Add lane departure warning
        if departure_direction:
            warning_text = f"Lane Departure Warning: {departure_direction.upper()}"
            cv2.putText(result, warning_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            socketio.emit('play_beep', {'direction': departure_direction})
        
        # Encode the processed frame
        _, buffer = cv2.imencode('.jpg', result)
        processed_frame = base64.b64encode(buffer).decode('utf-8')
        
        return processed_frame
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return None

if __name__ == '__main__':
    try:
        socketio.start_background_task(process_email_queue)
        print("Starting DriveAI on http://0.0.0.0:3000")
        socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 3000)), debug=True, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting server: {e}") 