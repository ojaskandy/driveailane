#!/usr/bin/env python3
"""
Convert YOLO model to TensorFlow.js format

This script converts a PyTorch YOLO model to TensorFlow.js format
for use in the browser. It follows these steps:
1. Export the model to ONNX format
2. Convert ONNX to TensorFlow SavedModel
3. Convert SavedModel to TensorFlow.js

Requirements:
pip install torch onnx onnx-tf tensorflow tensorflowjs ultralytics
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def convert_yolo_to_tfjs(model_path, output_dir):
    """
    Convert a YOLO model to TensorFlow.js format
    
    Args:
        model_path: Path to the YOLO .pt model file
        output_dir: Directory to save the converted model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Export YOLO model to ONNX
    logger.info(f"Exporting YOLO model to ONNX format from {model_path}")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        onnx_path = str(Path(output_dir) / "model.onnx")
        model.export(format="onnx", opset=12, simplify=True)
        # Move the exported model to the specified location
        exported_onnx = Path(model_path).with_suffix('.onnx')
        if exported_onnx.exists():
            os.rename(exported_onnx, onnx_path)
            logger.info(f"ONNX model saved to {onnx_path}")
        else:
            logger.error(f"ONNX export failed. File not found at {exported_onnx}")
            return False
    except Exception as e:
        logger.error(f"ONNX export failed: {str(e)}")
        return False
    
    # Step 2: Convert ONNX to TensorFlow SavedModel
    logger.info("Converting ONNX model to TensorFlow SavedModel")
    try:
        import onnx
        from onnx_tf.backend import prepare
        
        # Load the ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TensorFlow
        tf_model_dir = str(Path(output_dir) / "tf_model")
        os.makedirs(tf_model_dir, exist_ok=True)
        
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(tf_model_dir)
        logger.info(f"TensorFlow model saved to {tf_model_dir}")
    except Exception as e:
        logger.error(f"TensorFlow conversion failed: {str(e)}")
        return False
    
    # Step 3: Convert SavedModel to TensorFlow.js
    logger.info("Converting TensorFlow SavedModel to TensorFlow.js")
    try:
        import tensorflowjs as tfjs
        
        tfjs_model_dir = str(Path(output_dir) / "tfjs_model")
        os.makedirs(tfjs_model_dir, exist_ok=True)
        
        tfjs.converters.convert_tf_saved_model(
            tf_model_dir,
            tfjs_model_dir
        )
        logger.info(f"TensorFlow.js model saved to {tfjs_model_dir}")
        
        # Step 4: Copy model to web directory for serving
        web_model_dir = "drivesafe/web/static/models/tfjs_model"
        os.makedirs(web_model_dir, exist_ok=True)
        
        # Copy all files from tfjs_model_dir to web_model_dir
        for file in os.listdir(tfjs_model_dir):
            source = os.path.join(tfjs_model_dir, file)
            destination = os.path.join(web_model_dir, file)
            if os.path.isfile(source):
                shutil.copy2(source, destination)
        
        logger.info(f"Copied TensorFlow.js model to {web_model_dir} for web serving")
    except Exception as e:
        logger.error(f"TensorFlow.js conversion or copy failed: {str(e)}")
        return False
    
    logger.info("Conversion complete! The TensorFlow.js model is ready for use in the browser.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO model to TensorFlow.js format")
    parser.add_argument("--model", required=True, help="Path to the YOLO .pt model file")
    parser.add_argument("--output", default="converted_model", help="Directory to save the converted model")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return 1
    
    success = convert_yolo_to_tfjs(args.model, args.output)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 