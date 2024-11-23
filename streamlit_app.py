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
    """
    Load and configure YOLO model from Hugging Face Hub with optimizations.
    Returns:
        YOLO model instance or None if loading fails
    """
    try:
        # Set up torch configurations for CPU optimization
        torch.set_grad_enabled(False)  # Disable gradient computation
        if torch.backends.mkl.is_available():
            torch.backends.mkl.enabled = True
        
        # Configure torch for inference optimization
        torch.set_num_threads(4)  # Adjust based on your CPU cores
        
        # Download model with error handling and timeout
        try:
            model_path = hf_hub_download(
                repo_id="oliverlibaw/fred-george-gary-11-2024.pt",
                filename="cats_yolov8n_11-21-v2.pt",
                cache_dir="model_cache",
                force_download=False,  # Use cached version if available
                resume_download=True,  # Resume interrupted downloads
                token=None,  # Add your HF token here if model is private
                timeout=60  # 60 second timeout
            )
        except Exception as download_err:
            st.error(f"Failed to download model: {str(download_err)}")
            return None
            
        # Load and configure model
        model = YOLO(model_path, task='detect')
        
        # Model configurations
        model.conf = 0.25  # Default confidence threshold
        model.iou = 0.45  # Default IOU threshold
        
        # Set model to evaluation mode
        model.model.eval()
        
        # Force model to CPU mode since we're not using GPU
        model.model.cpu()
        
        # Clear CUDA cache if it was previously used
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Log successful model loading
        st.success("Model loaded successfully!")
        
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        # Log detailed error information
        st.error(f"Error type: {type(e).__name__}")
        st.error(f"Error details: {str(e)}")
        return None
def process_image(image, model, conf_threshold=0.25, target_size=(640, 640)):
    """Process image with optimized resizing for both small and large images"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        original_size = image.size
        
        # Only resize if image is extremely large
        max_dim = max(original_size)
        if max_dim > 4096:  # Much more conservative resize threshold
            scale_factor = 4096 / max_dim
            new_size = (int(original_size[0] * scale_factor),
                       int(original_size[1] * scale_factor))
            processed_image = image.resize(new_size, Image.Resampling.LANCZOS)
            st.write(f"Resized from {original_size} to {new_size}")
        else:
            processed_image = image
            st.write(f"Using original size: {original_size}")
        
        # Process image
        if resize_needed:
            processed_image = image.resize(new_size, Image.Resampling.LANCZOS)
            # Log resize operation
            st.write(f"Resized from {original_size} to {new_size}")
        else:
            processed_image = image
            st.write(f"Using original size: {original_size}")
            
        # Convert to numpy array
        image_np = np.array(processed_image)
        
        # Run inference
        results = model(image_np, conf=conf_threshold, verbose=False)[0]
        
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
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        quality = st.select_slider(
            "Image Quality",
            options=["Standard", "High", "Ultra"],
            value="High"
        )
        
        quality_sizes = {
            "Standard": (1024, 1024),
            "High": (1280, 1280),
            "Ultra": (1536, 1536)
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