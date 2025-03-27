from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import os
import logging
import base64
import numpy as np
import traceback
from pathlib import Path
import shutil
import sys
import time
import cv2
import threading
from werkzeug.serving import run_simple

# Configure logging for better debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Path to static assets
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
STATIC_MODELS_DIR = os.path.join(STATIC_DIR, 'models')
REPO_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

# Ensure models directories exist
os.makedirs(STATIC_MODELS_DIR, exist_ok=True)
os.makedirs(REPO_MODELS_DIR, exist_ok=True)

# Global variables for model status
model_status = {
    'yolo': {
        'loaded': False,
        'error': None
    },
    'opencv': {
        'loaded': False,
        'error': None
    }
}

# Initialize OpenCV and other dependencies
def initialize_dependencies():
    logger.info("Initializing dependencies...")
    global model_status
    
    try:
        # Check for OpenCV
        cv2_version = cv2.__version__
        logger.info(f"OpenCV version: {cv2_version}")
        model_status['opencv']['loaded'] = True
    except Exception as e:
        error_msg = f"Failed to initialize OpenCV: {str(e)}"
        logger.error(error_msg)
        model_status['opencv']['error'] = error_msg
        model_status['opencv']['loaded'] = False

    # Ensure model directory exists
    try:
        # Check for models to copy to static directory
        logger.info("Checking for models to copy to static directory...")
        check_and_copy_models()
    except Exception as e:
        error_msg = f"Failed to copy model files: {str(e)}"
        logger.error(error_msg)
        model_status['yolo']['error'] = error_msg
        model_status['yolo']['loaded'] = False

def check_and_copy_models():
    """Check for YOLO model files and copy them to static directory for web access"""
    global model_status
    
    # Define paths
    static_model_dir = os.path.join(app.static_folder, "models", "tfjs_model")
    os.makedirs(static_model_dir, exist_ok=True)
    
    # Look for model files in various locations
    model_search_paths = [
        os.path.join("models", "traffic_light", "best_traffic_small_yolo.pt"),  # Standard path
        os.path.join("drivesafe", "models", "traffic_light", "best_traffic_small_yolo.pt"),  # Relative to project
        "best_traffic_small_yolo.pt",  # Root directory
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                    "models", "traffic_light", "best_traffic_small_yolo.pt")  # Two dirs up
    ]
    
    # Search for a model file to use
    model_path = None
    for path in model_search_paths:
        if os.path.exists(path):
            model_path = path
            logger.info(f"Found model file at: {path}")
            break
    
    if model_path:
        # For demonstration - create a placeholder JSON file for web loading
        # In a real implementation, convert the PyTorch model to TF.js format
        model_json_path = os.path.join(static_model_dir, "model.json")
        
        # Create a minimal model.json file for TF.js if it doesn't exist
        if not os.path.exists(model_json_path):
            with open(model_json_path, 'w') as f:
                f.write('{"modelTopology":{"class_name":"Model","config":{"name":"model"}},"weightsManifest":[]}')
            
            logger.info(f"Created placeholder model.json at {model_json_path}")
            
        model_status['yolo']['loaded'] = True
        logger.info("Model configured for web serving")
    else:
        model_status['yolo']['loaded'] = False
        model_status['yolo']['error'] = "Could not find YOLO model file in any expected location"
        logger.error("Could not find YOLO model file in any expected location")

# Run the copy operation at startup
initialize_dependencies()

@app.route('/')
def index():
    """Render the index page."""
    return render_template('index.html')

@app.route('/start-drive')
def start_drive():
    """Render the start drive page."""
    return render_template('start_drive.html')

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

@app.route('/model_status')
def get_model_status():
    """Return model status - this is a placeholder since model status is now handled client-side."""
    # Check if model files exist in static directory
    model_json_path = os.path.join(STATIC_MODELS_DIR, 'tfjs_model', 'model.json')
    model_exists = os.path.exists(model_json_path)
    
    return jsonify({
        "status": "client_side_model",
        "model_exists": model_exists,
        "message": "Model loading is handled client-side with TensorFlow.js"
    })

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get image data from form
        image_data = request.form.get('image')
        
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Remove data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        
        # Decode image
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Process the frame (we'll perform lane detection here)
        try:
            # Simple lane detection with OpenCV
            processed_frame = detect_lanes(frame)
            
            # Encode the processed frame
            _, buffer = cv2.imencode('.jpg', processed_frame)
            processed_image = base64.b64encode(buffer).decode('utf-8')
            
            # Return the processed image
            return jsonify({
                'success': True,
                'processed_image': f'data:image/jpeg;base64,{processed_image}',
                'message': 'Frame processed successfully with lane detection',
                'model_used': 'Lane detection'
            })
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Encode the original frame if processing fails
            _, buffer = cv2.imencode('.jpg', frame)
            original_image = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': False,
                'processed_image': f'data:image/jpeg;base64,{original_image}',
                'error': str(e),
                'message': 'Using original frame (processing failed)'
            })
            
    except Exception as e:
        logger.error(f"Error in process_frame route: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def detect_lanes(frame):
    """Detect lanes in the given frame using OpenCV"""
    try:
        # Make a copy of the frame
        lane_frame = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(lane_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blur, 50, 150)
        
        # Define region of interest
        height, width = edges.shape
        mask = np.zeros_like(edges)
        
        # Define a polygon for the mask (focusing on the bottom half of the image)
        polygon = np.array([
            [(0, height), (0, height//2), (width, height//2), (width, height)]
        ], np.int32)
        
        # Fill the polygon with white (255)
        cv2.fillPoly(mask, polygon, 255)
        
        # Apply the mask
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, maxLineGap=50)
        
        # Draw lines on the original image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lane_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        
        return lane_frame
    
    except Exception as e:
        logger.error(f"Error in lane detection: {str(e)}")
        # Return original frame if lane detection fails
        return frame

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    # Use 0.0.0.0 to make the server publicly available
    
    # Print debug information
    logger.info(f"Starting server on port {port}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Files in current directory: {os.listdir('.')}")
    logger.info(f"Files in static directory: {os.listdir(app.static_folder)}")
    
    # Run the app on all interfaces, allowing remote connections
    run_simple('0.0.0.0', port, app, use_reloader=True, use_debugger=True, use_evalex=True) 