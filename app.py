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

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
thread_lock = Lock()

# Queue for storing email addresses
email_queue = queue.Queue()

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

def send_email(email):
    try:
        # Add email to queue for batch processing
        email_queue.put(email)
        return True
    except Exception as e:
        print(f"Error queueing email: {str(e)}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

@app.route('/join-waitlist', methods=['POST'])
def join_waitlist():
    try:
        data = request.get_json()
        email = data.get('email')
        if email:
            if send_email(email):
                return jsonify({'status': 'success'})
        return jsonify({'status': 'error', 'message': 'Invalid email'}), 400
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')

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

def process_email_queue():
    """Process emails in batches and send to your email"""
    while True:
        try:
            emails = []
            # Collect all available emails from the queue
            while not email_queue.empty() and len(emails) < 100:  # Process up to 100 emails at a time
                emails.append(email_queue.get_nowait())
            
            if emails:
                # Create email content
                email_content = "\n".join(emails)
                msg = MIMEText(f"New waitlist signups:\n\n{email_content}")
                msg['Subject'] = 'DriveAI Waitlist Signups'
                msg['From'] = 'noreply@driveai.com'
                msg['To'] = 'ojaskandy@gmail.com'
                
                # Send email (commented out for now)
                # with smtplib.SMTP('smtp.gmail.com', 587) as server:
                #     server.starttls()
                #     server.login('your-email@gmail.com', 'your-password')
                #     server.send_message(msg)
                
                # For now, just print the emails
                print(f"Would send emails: {email_content}")
            
            socketio.sleep(300)  # Process every 5 minutes
            
        except Exception as e:
            print(f"Error processing email queue: {str(e)}")
            socketio.sleep(60)  # Wait a minute before retrying on error

if __name__ == '__main__':
    socketio.start_background_task(process_email_queue)
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 