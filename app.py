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
def handle_frame(frame_data):
    try:
        # Ensure we're getting just the base64 data
        if isinstance(frame_data, str):
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
        
        # Decode the base64 image
        nparr = np.frombuffer(base64.b64decode(frame_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("Failed to decode frame")
            return None
            
        # Process the frame
        processed_frame = process_frame(frame)
        if processed_frame:
            # Ensure we're sending just the base64 data without the data URL prefix
            if processed_frame.startswith('data:image/jpeg;base64,'):
                processed_frame = processed_frame.split(',')[1]
            emit('processed_frame', processed_frame)
    except Exception as e:
        print(f"Error handling frame: {str(e)}")

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
        
        # Define dashboard area (yellow rectangle) dimensions
        dashboard_height = int(height * 0.25)  # Bottom 25% of frame is dashboard
        dashboard_y = height - dashboard_height  # Y coordinate for top of dashboard
        
        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for white and yellow lanes (adjusted for real-world conditions)
        # White is now more tolerant of different lighting conditions
        white_lower = np.array([0, 0, 160])  # Lowered value threshold
        white_upper = np.array([180, 45, 255])  # Increased saturation threshold
        
        # Yellow is now more tolerant of faded lines
        yellow_lower = np.array([15, 40, 100])  # Adjusted for faded yellow
        yellow_upper = np.array([35, 255, 255])  # Wider hue range
        
        # Create masks for white and yellow lanes
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Apply morphological operations to enhance lane markings
        kernel = np.ones((3,3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Create dashboard mask to exclude dashboard area from lane detection
        dashboard_mask = np.ones_like(combined_mask)
        dashboard_mask[dashboard_y:, :] = 0  # Mask out dashboard area
        
        # Apply dashboard mask to combined mask
        combined_mask = cv2.bitwise_and(combined_mask, dashboard_mask)
        
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(combined_mask, (5, 5), 0)
        
        # Apply Canny edge detection with optimized parameters
        edges = cv2.Canny(blur, 30, 150)  # Lowered minimum threshold
        
        # Define region of interest - make it more focused on the road
        roi_vertices = np.array([[
            (0, height),                    # Bottom left
            (0, dashboard_y),               # Top left
            (width, dashboard_y),           # Top right
            (width, height)                 # Bottom right
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
            threshold=25,            # Lowered threshold for better detection
            minLineLength=40,        # Shortened minimum line length
            maxLineGap=150          # Increased max gap to connect broken lines
        )
        
        # Draw lines on the original frame
        line_image = np.zeros_like(frame)
        if lines is not None:
            # Separate left and right lane lines
            left_lines = []
            right_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Skip if line is in dashboard area
                if y1 > dashboard_y or y2 > dashboard_y:
                    continue
                    
                if x2 - x1 == 0:  # Skip vertical lines
                    continue
                    
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter lines by slope and position (adjusted for real-world conditions)
                if 0.2 < abs(slope) < 2.5:  # Wider slope range
                    if slope < 0 and x1 < width/2:  # Left lane
                        left_lines.append(line)
                    elif slope > 0 and x1 > width/2:  # Right lane
                        right_lines.append(line)
            
            # Function to average and extrapolate lines
            def average_lines(lines):
                if not lines:
                    return None
                    
                avg_line = np.mean(lines, axis=0, dtype=np.int32)
                x1, y1, x2, y2 = avg_line[0]
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                
                # Extrapolate to bottom of frame (just above dashboard)
                y_bottom = dashboard_y
                x_bottom = int(x1 + (y_bottom - y1) / slope) if slope != 0 else x1
                
                # Extrapolate to top of ROI
                y_top = int(height * 0.45)  # Extended higher up
                x_top = int(x1 + (y_top - y1) / slope) if slope != 0 else x1
                
                return np.array([[x_top, y_top, x_bottom, y_bottom]])
            
            # Get averaged lines
            left_line = average_lines(left_lines)
            right_line = average_lines(right_lines)
            
            # Draw the averaged lines
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
        
        # Combine the original frame with the line image
        result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
        
        # Draw dashboard rectangle (yellow) - full width
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
            
            # Play beep sound
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