import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
from huggingface_hub import hf_hub_download

# Set page config
st.set_page_config(page_title="Object Detection App", layout="wide")

# Get model from HF - this should be outside the main() function
@st.cache_resource
def load_model():
    try:
        # Download model from HF Hub
        model_path = hf_hub_download(
            repo_id="oliverlibaw/",  # repo name should not include .pt
            filename="yolov8scats112024",
        )
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    st.title("Object Detection with YOLOv8")
    st.write("Upload an image to detect objects")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check your Hugging Face repository settings.")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Convert uploaded file to image
        image = Image.open(uploaded_file)
        
        # Create two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Original Image")
            st.image(image, use_column_width=True)
        
        if st.button('Detect Objects'):
            with st.spinner('Detecting objects...'):
                # Perform inference
                results = model(image)
                
                # Plot results
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    img = result.plot()
                    
                    with col2:
                        st.write("Detection Results")
                        st.image(img, use_column_width=True)
                        
                        # Display detection information
                        st.write("Detections:")
                        for box in boxes:
                            conf = box.conf[0]
                            cls = int(box.cls[0])
                            cls_name = model.names[cls]
                            st.write(f"- {cls_name}: {conf:.2f}")
             

if __name__ == '__main__':
    main()