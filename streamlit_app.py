import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
import time
import os

# Set page config
st.set_page_config(
    page_title="Cat Detection App",
    page_icon="🐱",
    layout="wide"
)

# Cache the model loading
@st.cache_resource
def load_model():
    """Load model from Hugging Face Hub"""
    try:
        model_path = hf_hub_download(
            repo_id="oliverlibaw/fred-george-gary-11-2024.pt",
            filename="cats_yolov8n_11-21-v2.pt",
            cache_dir="model_cache",
            force_download=True  # Added to ensure fresh download
        )
        model = YOLO(model_path, task='detect')
        model.model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model, conf_threshold=0.25, nms_iou=0.45, target_size=(640, 640)):
    """Process image with optimized resizing for both small and large images"""
    try:
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Store original size
        original_size = image.size
        
        # Determine if we need to resize
        min_dim = min(original_size)
        max_dim = max(original_size)
        
        # Calculate resize factor
        resize_needed = False
        
        # For images smaller than target size, scale up to 640 exactly
        if min_dim < target_size[0]:
            scale_factor = target_size[0] / min_dim
            new_size = (int(original_size[0] * scale_factor), 
                       int(original_size[1] * scale_factor))
            resize_needed = True
            
        # For large images, scale down if significantly larger than target
        elif max_dim > target_size[0] * 2:  # If more than 2x target size
            scale_factor = target_size[0] * 2 / max_dim
            new_size = (int(original_size[0] * scale_factor),
                       int(original_size[1] * scale_factor))
            resize_needed = True
        
        # Process image
        if resize_needed:
            processed_image = image.resize(new_size, Image.Resampling.LANCZOS)
            st.write(f"Resized from {original_size} to {new_size}")
        else:
            processed_image = image
            st.write(f"Using original size: {original_size}")
            
        # Convert to numpy array
        image_np = np.array(processed_image)
        
        # Run inference with NMS
        results = model(image_np, conf=conf_threshold, iou=nms_iou, verbose=False)[0]
        
        # Convert back to original size for visualization if we resized
        if resize_needed:
            image_np = np.array(image)
            
        detections = []
        # Process detections
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Scale coordinates back if we resized
            if resize_needed:
                x1 = int(x1 * original_size[0] / new_size[0])
                x2 = int(x2 * original_size[0] / new_size[0])
                y1 = int(y1 * original_size[1] / new_size[1])
                y2 = int(y2 * original_size[1] / new_size[1])
            
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            class_name = results.names[cls]
            
            # Color based on confidence
            color = (0, 255, 0) if conf > 0.8 else (0, 255, 255) if conf > 0.5 else (0, 0, 255)
            
            # Draw box
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{class_name}: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image_np, (x1, y1-h-10), (x1+w, y1), color, -1)
            cv2.putText(image_np, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            detections.append({
                'class': class_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })
        
        return image_np, detections
        
    except Exception as e:
        st.error(f"Error in image processing: {str(e)}")
        return None, []

def main():
    st.title("Cat Detection App")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please refresh the page.")
        st.stop()
        
    # Settings sidebar
    with st.sidebar:
        st.header("Detection Settings")
        
        # Confidence threshold slider
        conf_threshold = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Adjust the minimum confidence score required for a detection to be shown"
        )
        
        # NMS IoU threshold slider
        nms_iou = st.slider(
            "NMS IoU Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="Adjust the Intersection over Union threshold for Non-Maximum Suppression"
        )
        
        # Image quality selector
        quality = st.select_slider(
            "Image Quality",
            options=["Standard", "High", "Ultra"],
            value="High",
            help="Higher quality may improve detection but will be slower"
        )
        
        quality_sizes = {
            "Standard": (640, 640),    # Matches training size
            "High": (960, 960),        # 1.5x training size
            "Ultra": (1280, 1280)      # 2x training size
        }

    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Camera Input")
        camera_image = st.camera_input("Take a picture")
        uploaded_file = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])
        
    with col2:
        st.subheader("Detection Results")
        input_image = camera_image or uploaded_file
        
        if input_image:
            try:
                start_time = time.time()
                image = Image.open(input_image)
                
                # Process and show results
                annotated_img, detections = process_image(
                    image,
                    model,
                    conf_threshold,
                    nms_iou,
                    quality_sizes[quality]
                )
                
                if annotated_img is not None:
                    st.image(annotated_img, channels="RGB", use_column_width=True)
                    
                    if detections:
                        st.write("Detections:")
                        for det in detections:
                            conf = det['confidence']
                            color = 'green' if conf > 0.8 else 'orange' if conf > 0.5 else 'red'
                            st.markdown(f"- {det['class']}: ::{color}[{conf:.2f}]")
                    else:
                        st.info("No cats detected in image")
                    
                    # Show processing info
                    process_time = time.time() - start_time
                    st.write(f"Processing Time: {process_time:.3f}s")
                    st.write(f"Image Size: {image.size}")
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

if __name__ == '__main__':
    main()