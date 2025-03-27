#!/usr/bin/env python3
"""
DriveSafe Application Runner

This script simplifies running the DriveSafe application by:
1. Checking if the model is already converted
2. Converting the model if needed (if a .pt file is found)
3. Starting the Flask application

Usage:
    python run_app.py [--port PORT]
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def find_model():
    """Find a YOLO model file (.pt) in the models directory."""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    
    # Look for .pt files
    for file in os.listdir(models_dir):
        if file.endswith('.pt'):
            return os.path.join(models_dir, file)
    
    return None

def check_converted_model():
    """Check if a converted TensorFlow.js model exists."""
    static_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                    'drivesafe', 'web', 'static', 'models', 'tfjs_model')
    
    if os.path.exists(static_models_dir):
        model_json = os.path.join(static_models_dir, 'model.json')
        if os.path.exists(model_json):
            return True
    
    return False

def convert_model(model_path):
    """Convert the YOLO model to TensorFlow.js format."""
    logger.info(f"Converting model: {model_path}")
    
    try:
        # Run the conversion script
        convert_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'convert_model.py')
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'converted_model')
        
        result = subprocess.run([
            sys.executable, 
            convert_script, 
            '--model', model_path,
            '--output', output_dir
        ], check=True, capture_output=True, text=True)
        
        logger.info("Model conversion completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Model conversion failed: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error during model conversion: {e}")
        return False

def run_flask_app(port=5000):
    """Run the Flask application."""
    logger.info(f"Starting Flask app on port {port}")
    
    try:
        app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              'drivesafe', 'web', 'app.py')
        
        # Set the port environment variable
        env = os.environ.copy()
        env['PORT'] = str(port)
        
        # Run the Flask app
        subprocess.run([sys.executable, app_path], env=env)
        return True
    except Exception as e:
        logger.error(f"Error starting Flask app: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the DriveSafe application")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the Flask app on")
    parser.add_argument("--skip-convert", action="store_true", help="Skip model conversion")
    
    args = parser.parse_args()
    
    # Check if model is already converted
    if not check_converted_model() and not args.skip_convert:
        logger.info("No converted model found. Looking for .pt model file...")
        
        # Find a .pt model file
        model_path = find_model()
        if model_path:
            logger.info(f"Found model: {model_path}")
            
            # Convert the model
            if not convert_model(model_path):
                logger.warning("Model conversion failed. The app will try to use the CDN fallback.")
        else:
            logger.warning("No .pt model file found. The app will try to use the CDN fallback.")
    else:
        if args.skip_convert:
            logger.info("Skipping model conversion as requested")
        else:
            logger.info("Converted model already exists")
    
    # Run the Flask app
    run_flask_app(args.port)

if __name__ == "__main__":
    main() 