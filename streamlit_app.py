import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image, ImageEnhance
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
import time
import os
import pygame
from pathlib import Path

# Set page config
st.set_page_config(page_title="Cat Detection App", layout="wide")

# Dictionary to store sound files for each class
SOUND_FILES = {
    'fred': 'fred.wav',
    'george': 'george.wav',
    'gary': 'gary.wav'
}

# Dictionary to track when each sound was last played
last_played = {}
MIN_TIME_BETWEEN_SOUNDS = 2.0

@st.cache_resource
def load_model():
    """Enhanced model loading with verification and warmup"""
    try:
        # Download model from HF Hub
        model_path = hf_hub_download(
            repo_id="oliverlibaw/fred-george-gary-11-2024.pt",
            filename="cats_yolov8n_11-21-v2.pt",
        )
        
        # Load and verify model
        model = YOLO(model_path)
        if not hasattr(model, 'names') or not model.names:
            raise ValueError("Model loaded but missing class names")
        
        # Force model to evaluation mode
        model.eval()
        
        # Warmup run
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy_input, verbose=False)
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model, conf_threshold=0.25, target_size=(640, 640)):
    """Enhanced image processing with better error handling and preprocessing"""
    try:
        # Convert PIL Image to RGB numpy array
        if isinstance(image, Image.Image):
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize while maintaining aspect ratio
            ratio = min(target_size[0] / image.size[0], target_size[1] / image.size[1])
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            image_np = np.array(image)
        else:
            raise ValueError("Input must be a PIL Image")

        # Ensure correct format and range
        if image_np.dtype != np.uint8:
            image_np = (image_np * 255).astype(np.uint8)
            
        # Store original size for scaling back
        orig_size = image_np.shape[:2]
        
        # Run inference
        results = model(image_np, conf=conf_threshold, verbose=False)[0]
        detections = []
        
        # Process detections
        for box in results.boxes:
            # Get box coordinates and scale to original size
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            class_name = results.names[cls]
            
            # Choose color based on confidence
            if conf > 0.8:
                color = (0, 255, 0)  # Green
            elif conf > 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
                
            # Draw rectangle
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{class_name}: {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image_np, (x1, y1-label_h-5), (x1+label_w, y1), color, -1)
            cv2.putText(image_np, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            detections.append({
                'class': class_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })
            
        return image_np, detections
        
    except Exception as e:
        st.error(f"Error in image processing: {str(e)}")
        return None, []

def initialize_audio():
    """Initialize audio system with error handling"""
    if 'audio_system' not in st.session_state:
        try:
            os.environ['SDL_AUDIODRIVER'] = 'dummy'
            pygame.mixer.init()
            st.session_state.audio_system = True
        except Exception as e:
            st.warning(f"Audio system initialization failed: {str(e)}")
            st.session_state.audio_system = False
    return st.session_state.audio_system

def simulate_sound(class_name):
    """Simulate sound playing with visual feedback"""
    st.toast(f"🔊 Detected {class_name}!")

def main():
    st.title("Cat Detection App")
    
    # Initialize systems
    audio_initialized = initialize_audio()
    model = load_model()
    
    if model is None:
        st.error("Failed to load model. Please refresh the page or contact support.")
        st.stop()

    # Sidebar settings
    with st.sidebar:
        st.header("Detection Settings")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Adjust detection sensitivity"
        )
        
        st.header("Image Settings")
        image_quality = st.select_slider(
            "Image Quality",
            options=["Low", "Medium", "High"],
            value="High",
            help="Higher quality may affect performance"
        )
        
        show_debug = st.checkbox("Show Debug Info", False)
        
        # Quality to size mapping
        quality_sizes = {
            "Low": (320, 320),
            "Medium": (640, 640),
            "High": (1280, 1280)
        }

    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Camera Input")
        camera_image = st.camera_input(
            "Take a picture",
            help="Click to capture an image"
        )
        
    with col2:
        st.subheader("Detection Results")
        if camera_image is not None:
            try:
                # Process image
                start_time = time.time()
                
                image = Image.open(camera_image)
                if image.size[0] < 320 or image.size[1] < 320:
                    st.warning("Image resolution is very low. This may affect detection quality.")
                
                # Process image with selected quality
                annotated_img, detections = process_image(
                    image, 
                    model, 
                    conf_threshold,
                    target_size=quality_sizes[image_quality]
                )
                
                if annotated_img is not None:
                    # Display results
                    st.image(annotated_img, channels="RGB", use_column_width=True)
                    
                    # Show detections
                    if detections:
                        st.write("Detections:")
                        for det in detections:
                            conf = det['confidence']
                            color = 'green' if conf > 0.8 else 'orange' if conf > 0.5 else 'red'
                            st.markdown(f"- {det['class']}: ::{color}[{conf:.2f}]")
                            simulate_sound(det['class'])
                    else:
                        st.info("No cats detected in image")
                    
                    # Show debug info
                    if show_debug:
                        process_time = time.time() - start_time
                        st.write("Debug Information:")
                        st.write(f"- Processing Time: {process_time:.3f} seconds")
                        st.write(f"- Image Size: {image.size}")
                        st.write(f"- Image Mode: {image.mode}")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.write("Please try taking another picture.")

    # Add helpful information in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("""
        1. Allow camera access when prompted
        2. Click 'Take a picture' to capture
        3. Adjust settings as needed
        4. Watch for detections and sounds
        """)
        
        st.markdown("### About")
        st.markdown("""
        This app detects three cats:
        - Fred
        - George
        - Gary
        
        Confidence colors:
        - 🟢 Green: High (>0.8)
        - 🟡 Yellow: Medium (>0.5)
        - 🔴 Red: Low (<0.5)
        """)

if __name__ == '__main__':
    main()