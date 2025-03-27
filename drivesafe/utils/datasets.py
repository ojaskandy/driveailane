import cv2
import numpy as np

def letterbox(img, new_shape=416):
    """
    Resize image to a 32-pixel multiple rectangle
    """
    # For testing, simply resize the image
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    return [cv2.resize(img, new_shape)] 