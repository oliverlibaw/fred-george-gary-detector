# utils.py
import cv2
import numpy as np
import torch

def preprocess_image(image):
    """Convert PIL Image to CV2 format"""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def draw_boxes(image, boxes, labels, colors):
    """Draw bounding boxes on image"""
    img = image.copy()
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box[:4])
        conf = box[4]
        color = colors[int(label)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f'{conf:.2f}', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return img