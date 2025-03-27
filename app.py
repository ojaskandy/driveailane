from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO
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
from datetime import datetime
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
thread_lock = Lock()

# Twilio credentials
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.environ.get('TWILIO_PHONE_NUMBER')
TARGET_PHONE_NUMBER = '+14256230692'
TARGET_EMAIL = 'ojaskandy@gmail.com'

# Queue for storing email addresses
email_queue = queue.Queue()

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
    print('Client connected')
    emit('stats_update', {
        'total_drives': trip_stats.total_drives,
        'total_time': format_duration(trip_stats.total_time),
        'last_drive': trip_stats.last_drive or 'Never'
    })

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('frame')
def handle_frame(data):
    try:
        with thread_lock:
            # Remove the data URL prefix
            if ',' not in data:
                return
            encoded_data = data.split(',')[1]
            
            # Decode base64 image
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            if nparr.size == 0:
                return
                
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return
                
            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))
            
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_frame = process(frame_rgb)
            
            if processed_frame is not None:
                # Encode processed frame
                processed_frame_encoded = encode_frame(processed_frame)
                if processed_frame_encoded:
                    # Send processed frame back to client
                    socketio.emit('processed_frame', processed_frame_encoded)
                    
    except Exception as e:
        print(f"Error processing frame: {str(e)}")

@socketio.on('start_trip')
def handle_start_trip(data):
    trip_id = str(datetime.now().timestamp())
    location = data.get('location', 'Location unavailable')
    
    trip_stats.current_trips[trip_id] = {
        'start_time': datetime.now(),
        'start_location': location,
        'current_location': location,
        'path': [location] if isinstance(location, dict) else [],
        'end_location': None
    }
    return {'trip_id': trip_id}

@socketio.on('update_location')
def handle_location_update(data):
    trip_id = data.get('trip_id')
    location = data.get('location')
    
    if trip_id and trip_id in trip_stats.current_trips:
        trip = trip_stats.current_trips[trip_id]
        trip['current_location'] = location
        if isinstance(location, dict):
            trip['path'].append(location)

@socketio.on('end_trip')
def handle_end_trip(data):
    trip_id = data.get('trip_id')
    location = data.get('location', 'Location unavailable')
    
    if trip_id and trip_id in trip_stats.current_trips:
        trip = trip_stats.current_trips[trip_id]
        end_time = datetime.now()
        duration = (end_time - trip['start_time']).total_seconds()
        
        # Update end location
        trip['end_location'] = location
        
        # Calculate trip statistics
        trip_stats.total_drives += 1
        trip_stats.total_time += duration
        trip_stats.last_drive = end_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Format locations for display
        start_loc = trip_stats.format_location(trip['start_location'])
        end_loc = trip_stats.format_location(location)
        
        # Prepare trip summary
        trip_summary = {
            'duration': format_duration(duration),
            'start_location': start_loc,
            'end_location': end_loc,
            'start_time': trip['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'path': trip['path'] if 'path' in trip else []
        }
        
        # Clean up
        del trip_stats.current_trips[trip_id]
        
        # Emit updated stats to all clients
        socketio.emit('stats_update', {
            'total_drives': trip_stats.total_drives,
            'total_time': format_duration(trip_stats.total_time),
            'last_drive': trip_stats.last_drive
        })
        
        return trip_summary
    return {'error': 'Invalid trip ID'}

def format_duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"

if __name__ == '__main__':
    socketio.start_background_task(process_email_queue)
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 