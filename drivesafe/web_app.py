"""
DriveSafe Web Application
Entry point for the web application when deployed on Render
"""

import os
import sys
from pathlib import Path

# Import the Flask app
try:
    from drivesafe.web.app import app
except ImportError:
    # Try relative import if the module structure is different
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from web.app import app

# This is needed for Render deployment
if __name__ == "__main__":
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get("PORT", 5000))
    # Start the application
    app.run(host="0.0.0.0", port=port)
