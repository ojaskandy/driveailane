import os

# Check if we're running in production (Render)
is_production = os.environ.get('RENDER', 'false').lower() == 'true'

if is_production:
    import eventlet
    eventlet.monkey_patch()

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
from PIL import Image
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
from lane_detector import LaneDetector

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Initialize SocketIO with appropriate async mode
if is_production:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
else:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

thread_lock = Lock()

# Create application context
app.app_context().push()

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

# Initialize lane detector
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
    return render_template('drive.html')

@app.route('/dashboard')
def dashboard():
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
        # Process frame using the same processing function for both environments
        processed_frame = process_frame(frame)
        if processed_frame:
            # Send processed frame back to client
            socketio.emit('processed_frame', processed_frame)

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
try:
    os.environ['SDL_AUDIODRIVER'] = 'dummy'  # Use dummy audio driver
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except Exception as e:
    print(f"Audio initialization failed: {str(e)}")
    AUDIO_AVAILABLE = False

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
    current_time = datetime.now()
    
    if left_pos is None and right_pos is None:
        # No lines detected
        if AUDIO_AVAILABLE and (current_time - last_beep_time).total_seconds() >= MIN_BEEP_INTERVAL:
            last_beep_time = current_time
            socketio.emit('play_beep', {'direction': 'no_lines'})
        return True, "No lane lines detected"
    
    # Calculate car position relative to detected lanes
    if left_pos is not None and right_pos is not None:
        car_pos = 0.5  # Assume car is in center of frame
        lane_center = (left_pos + right_pos) / 2
        offset = car_pos - lane_center
        
        # Check if car is too close to either lane
        if abs(offset) > LANE_DEPARTURE_THRESHOLD:
            if AUDIO_AVAILABLE and (current_time - last_beep_time).total_seconds() >= MIN_BEEP_INTERVAL:
                last_beep_time = current_time
                direction = "left" if offset > 0 else "right"
                socketio.emit('play_beep', {'direction': direction})
            return True, f"Lane Departure: {direction.upper()}"
    
    # If only one lane is detected, check distance to that lane
    elif left_pos is not None:
        if left_pos > LANE_DEPARTURE_THRESHOLD:
            if AUDIO_AVAILABLE and (current_time - last_beep_time).total_seconds() >= MIN_BEEP_INTERVAL:
                last_beep_time = current_time
                socketio.emit('play_beep', {'direction': 'left'})
            return True, "Lane Departure: LEFT"
    
    elif right_pos is not None:
        if right_pos < (1 - LANE_DEPARTURE_THRESHOLD):
            if AUDIO_AVAILABLE and (current_time - last_beep_time).total_seconds() >= MIN_BEEP_INTERVAL:
                last_beep_time = current_time
                socketio.emit('play_beep', {'direction': 'right'})
            return True, "Lane Departure: RIGHT"
    
    return False, None

def process_frame(frame):
    try:
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Define region of interest
        roi_vertices = np.array([[
            (0, height),
            (width//2, height//2),
            (width, height)
        ]], dtype=np.int32)
        
        # Apply ROI mask
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Apply Hough transform with optimized parameters
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=20,
            minLineLength=50,
            maxLineGap=100
        )
        
        # Create line image
        line_image = np.zeros_like(frame)
        
        if lines is not None:
            # Separate left and right lines
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
            
            # Draw left and right lines
            if left_lines:
                for line in left_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if right_lines:
                for line in right_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Combine original frame with line image
        result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
        
        # Check for lane departure
        left_pos, right_pos = calculate_lane_position(lines, width)
        departure_detected, departure_direction = check_lane_departure(left_pos, right_pos, width)
        
        # Add lane departure warning if detected
        if departure_detected:
            cv2.putText(result, departure_direction, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Encode the processed frame
        _, buffer = cv2.imencode('.jpg', result)
        processed_frame = base64.b64encode(buffer).decode('utf-8')
        
        return processed_frame
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return None

if __name__ == '__main__':
    # Get port from environment variable for deployment
    port = int(os.environ.get('PORT', 3000))
    
    # Check if we're running in production (Render)
    is_production = os.environ.get('RENDER', False)
    
    if is_production:
        # In production, use Gunicorn with eventlet
        from gunicorn.app.base import BaseApplication
        
        class StandaloneApplication(BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()
            
            def load_config(self):
                for key, value in self.options.items():
                    self.cfg.set(key, value)
            
            def load(self):
                return self.application
        
        options = {
            'bind': f'0.0.0.0:{port}',
            'worker_class': 'eventlet',
            'workers': 1,
            'timeout': 120
        }
        
        print(f"Starting DriveAI in production mode on port {port}")
        StandaloneApplication(app, options).run()
    else:
        # In development, use Flask's development server with threading
        print(f"Starting DriveAI in development mode on port {port}")
        socketio.run(app, host='0.0.0.0', port=port, debug=True, use_reloader=False) 