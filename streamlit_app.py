import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
from huggingface_hub import hf_hub_download
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

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

class VideoProcessor:
    def __init__(self, model):
        self.model = model
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Perform inference
        results = self.model(img, stream=True)
        
        # Process results and draw on frame
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            # Draw bounding boxes and labels
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cls_name = self.model.names[cls]
                
                # Draw rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label with confidence
                label = f"{cls_name}: {conf:.2f}"
                cv2.putText(img, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame.from_ndarray(img)

def main():
    st.title("Real-time Object Detection with YOLOv8")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check your Hugging Face repository settings.")
        st.stop()
    
    # RTC Configuration for WebRTC
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Create WebRTC streamer
    ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_configuration,
        video_processor_factory=lambda: VideoProcessor(model),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Add some instructions
    st.markdown("""
    ### Instructions:
    1. Click the 'Start' button above to begin the video stream
    2. Allow access to your camera when prompted
    3. The model will perform real-time detection on the video feed
    4. Objects will be highlighted with bounding boxes and labeled with confidence scores
    5. Click 'Stop' to end the stream
    """)
    
    # Add information about the model
    st.sidebar.header("Model Information")
    st.sidebar.write(f"Model: YOLOv8s")
    st.sidebar.write(f"Classes: {', '.join(model.names.values())}")

if __name__ == '__main__':
    main()