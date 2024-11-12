import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
from huggingface_hub import hf_hub_download
import time

# Set page config
st.set_page_config(page_title="Real-time Object Detection", layout="wide")

# Get model from HF - this should be outside the main() function
@st.cache_resource
def load_model():
    try:
        # Download model from HF Hub
        model_path = hf_hub_download(
            repo_id="oliverlibaw/catdetection",  # from your HF URL
            filename="yolov8scats1124.pt",       # your exact filename
        )
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.title("Real-time Object Detection with YOLOv8")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check your Hugging Face repository settings.")
        st.stop()

    # Create a placeholder for the video frame
    frame_placeholder = st.empty()
    
    # Add a stop button
    stop_button = st.button("Stop")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Unable to access camera. Please check your camera settings.")
        st.stop()
        
    try:
        while not stop_button:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera")
                break
                
            # Convert frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Perform detection
            results = model(frame_rgb, stream=True)
            
            # Process results and draw on frame
            for result in results:
                boxes = result.boxes.cpu().numpy()
                
                # Draw bounding boxes and labels
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cls_name = model.names[cls]
                    
                    # Draw rectangle
                    cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label with confidence
                    label = f"{cls_name}: {conf:.2f}"
                    cv2.putText(frame_rgb, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display the frame
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Add a small delay to reduce CPU usage
            time.sleep(0.1)
            
    finally:
        # Release the camera when done
        cap.release()
    
    # Add information about the model
    st.sidebar.header("Model Information")
    st.sidebar.write(f"Model: YOLOv8s")
    st.sidebar.write(f"Classes: {', '.join(model.names.values())}")

if __name__ == '__main__':
    main()