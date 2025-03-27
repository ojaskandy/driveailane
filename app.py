"""
DriveSafe Application
Root application entry point for Render deployment
"""

import os
import sys
from pathlib import Path

# Add the drivesafe directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the Flask app
from drivesafe.web.app import app

# This is used by Render to start the application
if __name__ == "__main__":
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get("PORT", 5000))
    # Start the application
    app.run(host="0.0.0.0", port=port) 