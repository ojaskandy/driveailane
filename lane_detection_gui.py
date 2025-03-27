import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def process(image):
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image,
                    np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    image_with_lines = draw_the_lines(image, lines)
    return image_with_lines

class LaneDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lane Detection")
        self.setGeometry(100, 100, 1200, 600)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # Create labels for raw and processed video
        self.raw_label = QLabel()
        self.processed_label = QLabel()
        
        # Add labels to layout
        layout.addWidget(self.raw_label)
        layout.addWidget(self.processed_label)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)  # Use 0 for default camera
        
        # Create timer for updating frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame for lane detection
            processed_frame = process(frame_rgb)
            
            # Convert frames to QImage
            raw_image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], 
                             frame_rgb.strides[0], QImage.Format_RGB888)
            processed_image = QImage(processed_frame.data, processed_frame.shape[1], 
                                   processed_frame.shape[0], processed_frame.strides[0], 
                                   QImage.Format_RGB888)
            
            # Scale images to fit labels
            raw_pixmap = QPixmap.fromImage(raw_image).scaled(self.raw_label.size(), 
                                                           Qt.KeepAspectRatio, 
                                                           Qt.SmoothTransformation)
            processed_pixmap = QPixmap.fromImage(processed_image).scaled(self.processed_label.size(), 
                                                                       Qt.KeepAspectRatio, 
                                                                       Qt.SmoothTransformation)
            
            # Display images
            self.raw_label.setPixmap(raw_pixmap)
            self.processed_label.setPixmap(processed_pixmap)
    
    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LaneDetectionGUI()
    window.show()
    sys.exit(app.exec_()) 