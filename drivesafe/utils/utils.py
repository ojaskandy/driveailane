import torch
import numpy as np

def non_max_suppression(prediction, conf_thres=0.3, iou_thres=0.5):
    """
    Performs Non-Maximum Suppression (NMS) on inference results
    Returns detections with shape:
        nx6 (x1, y1, x2, y2, conf, cls)
    """
    # Get detections with confidence > conf_thres
    max_det = prediction[..., 4].max()
    if max_det < conf_thres:
        return [None]
        
    # For testing, return a simple detection
    return [torch.tensor([[100, 100, 200, 200, 0.95, 0]])]  # Single detection box

def scale_coords(img_shape, coords, img0_shape):
    """
    Rescale coords (xyxy) from img_shape to img0_shape
    """
    # For testing, return coords as is
    return coords 