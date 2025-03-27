#!/usr/bin/env bash
set -o errexit

# Install system dependencies for OpenCV
echo "Installing system dependencies for OpenCV..."
apt-get update && apt-get install -y libsm6 libxrender1 libfontconfig1 libice6

# Ensure proper installation order with specific versions first
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directory structure..."
mkdir -p drivesafe/static/uploads drivesafe/static/detections drivesafe/models/traffic_light drivesafe/templates

# Model is now downloaded from Hugging Face at runtime
# No need to copy it during build

echo "Copying templates..."
cp -r templates/* drivesafe/templates/

echo "Copying static files..."
if [ -d "static" ]; then
    cp -r static/* drivesafe/static/
fi

echo "Final directory structure:"
ls -R drivesafe/
