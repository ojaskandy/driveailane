# Lane Detection Web Application

A real-time lane detection system that processes live camera feed to identify and highlight road lanes. This application is designed to be deployed on Render.

## Features

- Real-time camera feed processing
- Lane detection using OpenCV
- Side-by-side view of raw and processed feed
- Web-based interface
- Real-time updates using WebSocket

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/ojaskandy/driveailane.git
cd driveailane
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Deployment on Render

1. Push your code to GitHub
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Configure the service:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
   - Environment Variables:
     - `PYTHON_VERSION`: 3.9.0

## Technical Details

- Backend: Flask with Flask-SocketIO
- Frontend: HTML, CSS, JavaScript with Socket.IO client
- Image Processing: OpenCV
- Real-time Communication: WebSocket

## License

MIT License 