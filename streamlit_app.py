import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
from huggingface_hub import hf_hub_download

# Set page config
st.set_page_config(page_title="Real-time Object Detection", layout="wide")

# Get model from HF - this should be outside the main() function
@st.cache_resource
def load_model():
    try:
        # Download model from HF Hub
        model_path = hf_hub_download(
            repo_id="oliverlibaw/fred-george-gary-11-2024.pt",  # from your HF URL
            filename="best_11-19.pt",       # your exact filename
        )
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model, conf_threshold=0.25):
    """Process a single image and return the annotated image and detections"""
    # Convert PIL Image to numpy array if necessary
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Perform detection with confidence threshold
    results = model(image_np, conf=conf_threshold, stream=True)
    detections = []
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        
        # Draw bounding boxes and labels
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cls_name = model.names[cls]
            
            # Draw rectangle
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label with confidence
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(image_np, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detections.append({
                'class': cls_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })
    
    return image_np, detections

def main():
    st.title("Real-time Object Detection with YOLOv8")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please check your Hugging Face repository settings.")
        st.stop()

    # Add confidence threshold slider in sidebar
    st.sidebar.header("Detection Settings")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,  # default value
        step=0.05,
        help="Adjust the confidence threshold for object detection. Higher values mean stricter detection criteria."
    )

    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Camera Input")
        # Use Streamlit's camera input
        camera_image = st.camera_input("Take a picture")
        
    with col2:
        st.write("Detection Results")
        if camera_image is not None:
            # Convert the image to PIL Image
            image = Image.open(camera_image)
            
            # Process image and show results with confidence threshold
            annotated_img, detections = process_image(image, model, conf_threshold)
            
            # Display the annotated image
            st.image(annotated_img, channels="RGB", use_column_width=True)
            
            # Display detection information
            st.write("Detections:")
            for det in detections:
                st.write(f"- {det['class']}: {det['confidence']:.2f}")
    
    # Add information about the model in the sidebar
    st.sidebar.header("Model Information")
    st.sidebar.write(f"Model: YOLOv8s")
    st.sidebar.write(f"Classes: {', '.join(model.names.values())}")
    
    # Add instructions in the sidebar
    st.sidebar.header("Instructions")
    st.sidebar.write("""
    1. Allow camera access when prompted
    2. Click 'Take a picture' to capture an image
    3. Adjust the confidence threshold as needed
    4. The model will automatically detect objects
    5. View detection results and confidence scores
    """)

if __name__ == '__main__':
    main()