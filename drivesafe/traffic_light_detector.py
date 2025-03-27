import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch
from PIL import Image, ImageTk
import threading
from queue import Queue
import time
from datetime import datetime
from lane_detector import LaneDetector

class DriveSafeDetector:
    def __init__(self):
        # Initialize the GUI window
        self.root = tk.Tk()
        self.root.title("DriveSafe Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c2c2c')  # Dark theme
        
        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create output directories
        self.setup_directories()
        
        # Initialize models
        self.traffic_light_model = None
        self.lane_detector = LaneDetector()
        
        # Load the models
        self.setup_models()
        
        # Create GUI elements
        self.setup_gui()
        
        # Variables for video handling
        self.video_thread = None
        self.is_video_running = False
        self.frame_queue = Queue(maxsize=128)
        self.current_frame = None
        self.cap = None
        
        # Flag for clean shutdown
        self.is_closing = False
    
    def setup_directories(self):
        """Create necessary directories for outputs"""
        # Main directories
        self.models_dir = "models"
        self.output_dir = "outputs"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subdirectories for different models and outputs
        self.traffic_light_model_dir = os.path.join(self.models_dir, "traffic_light")
        self.traffic_light_output_dir = os.path.join(self.output_dir, "traffic_light_detections")
        self.lane_output_dir = os.path.join(self.output_dir, "lane_detections")
        
        os.makedirs(self.traffic_light_model_dir, exist_ok=True)
        os.makedirs(self.traffic_light_output_dir, exist_ok=True)
        os.makedirs(self.lane_output_dir, exist_ok=True)

    def setup_models(self):
        """Load traffic light detection model"""
        self.setup_traffic_light_model()
        print("Lane detection initialized")
    
    def setup_traffic_light_model(self):
        """Load the traffic light detection model"""
        print("Loading traffic light detection model...")
        try:
            model_path = os.path.join(self.traffic_light_model_dir, "best_traffic_small_yolo.pt")
            if not os.path.exists(model_path):
                messagebox.showerror("Error", 
                    "Traffic light model not found. Please place 'best_traffic_small_yolo.pt' in the models/traffic_light directory.")
                self.root.destroy()
                return
            self.traffic_light_model = YOLO(model_path)
            print("Traffic light model loaded successfully!")
        except Exception as e:
            print(f"Error loading traffic light model: {e}")
            messagebox.showerror("Error", f"Failed to load traffic light model: {e}")
            self.root.destroy()
            return
    
    def process_frame_for_lanes(self, frame):
        """Process a frame for lane detection"""
        return self.lane_detector.process_frame(frame)

    def process_video_feed(self):
        """Process video feed with both traffic light and lane detection"""
        while self.is_video_running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Mirror frame for webcam
                frame = cv2.flip(frame, 1)
                
                # Process frame with YOLO for traffic lights
                results = self.traffic_light_model(frame, conf=0.25)
                
                # Process results
                processed_frame = frame.copy()
                detections = {'red': 0, 'yellow': 0, 'green': 0}
                
                # Process traffic light detections
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Get class name and color
                        class_name = r.names[cls].lower()
                        
                        # Update detection counts
                        if 'red' in class_name:
                            detections['red'] += 1
                            color = (0, 0, 255)  # BGR
                        elif 'yellow' in class_name:
                            detections['yellow'] += 1
                            color = (0, 255, 255)
                        elif 'green' in class_name:
                            detections['green'] += 1
                            color = (0, 255, 0)
                        
                        # Draw detection box
                        cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(processed_frame, f'{class_name} {conf:.2f}',
                                  (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Process frame for lane detection
                processed_frame = self.process_frame_for_lanes(processed_frame)
                
                # Update stats
                self.root.after(0, lambda: self.update_stats_labels(
                    detections['red'],
                    detections['yellow'],
                    detections['green']
                ))
                
                # Convert to RGB for PIL
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                image = Image.fromarray(frame_rgb)
                
                # Resize while maintaining aspect ratio
                display_size = (800, 600)
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image)
                
                # Update video label
                self.video_label.configure(image=photo)
                self.video_label.image = photo
                
                # Save frame if detections found
                if sum(detections.values()) > 0:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_path = os.path.join(self.traffic_light_output_dir, f"detection_{timestamp}.jpg")
                    cv2.imwrite(output_path, processed_frame)
            else:
                self.stop_detection()
                break
            
            time.sleep(0.01)  # Small delay to prevent overwhelming the queue

    def on_closing(self):
        self.is_closing = True
        self.stop_detection()
        self.root.destroy()

    def setup_gui(self):
        """Setup the GUI elements"""
        # Main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video display
        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Control buttons frame
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=10)
        
        # Buttons
        self.camera_button = ttk.Button(self.control_frame, text="Start Camera", command=self.toggle_camera)
        self.camera_button.pack(side=tk.LEFT, padx=5)
        
        self.file_button = ttk.Button(self.control_frame, text="Open Video File", command=self.open_video)
        self.file_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(self.control_frame, text="Stop", command=self.stop_detection)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Statistics panel
        self.stats_frame = ttk.LabelFrame(self.main_frame, text="Detection Statistics", padding=10)
        self.stats_frame.pack(fill=tk.X, pady=10)
        
        # Traffic light stats
        self.red_label = ttk.Label(self.stats_frame, text="Red: 0")
        self.red_label.pack(side=tk.LEFT, padx=10)
        
        self.yellow_label = ttk.Label(self.stats_frame, text="Yellow: 0")
        self.yellow_label.pack(side=tk.LEFT, padx=10)
        
        self.green_label = ttk.Label(self.stats_frame, text="Green: 0")
        self.green_label.pack(side=tk.LEFT, padx=10)
        
        # Status label
        self.status_label = ttk.Label(self.main_frame, text="Ready")
        self.status_label.pack(pady=5)
    
    def toggle_camera(self):
        if not self.is_video_running:
            self.start_camera()
        else:
            self.stop_detection()
    
    def start_camera(self):
        if self.is_video_running:
            return
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_video_running = True
            self.video_thread = threading.Thread(target=self.process_video_feed)
            self.video_thread.daemon = True
            self.video_thread.start()
            
            self.status_label.config(text="Camera started")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {e}")
            self.stop_detection()
    
    def open_video(self):
        if self.is_video_running:
            self.stop_detection()
        
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.cap = cv2.VideoCapture(file_path)
                if not self.cap.isOpened():
                    raise Exception("Could not open video file")
                
                self.is_video_running = True
                self.video_thread = threading.Thread(target=self.process_video_feed)
                self.video_thread.daemon = True
                self.video_thread.start()
                
                self.status_label.config(text=f"Playing: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open video: {e}")
                self.stop_detection()
    
    def update_stats_labels(self, red_lights, yellow_lights, green_lights):
        self.red_label.config(text=f"Red: {red_lights}")
        self.yellow_label.config(text=f"Yellow: {yellow_lights}")
        self.green_label.config(text=f"Green: {green_lights}")
    
    def stop_detection(self):
        self.is_video_running = False
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        
        if not self.is_closing:
            self.status_label.config(text="Stopped")
            self.video_label.configure(image='')
            
            # Reset stats
            self.red_label.config(text="Red: 0")
            self.yellow_label.config(text="Yellow: 0")
            self.green_label.config(text="Green: 0")
    
    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.stop_detection()

if __name__ == "__main__":
    app = DriveSafeDetector()
    app.run() 