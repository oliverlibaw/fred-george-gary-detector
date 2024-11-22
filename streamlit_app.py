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
st.set_page_config(
    page_title="Cat Detection App",
    page_icon="🐱",
    layout="wide"
)
# Ensure cache directory exists
os.makedirs("model_cache", exist_ok=True)

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
    """Load model from Hugging Face Hub with task specification"""
    try:
        # Download model from HF Hub
        model_path = hf_hub_download(
            repo_id="oliverlibaw/fred-george-gary-11-2024.pt",
            filename="cats_yolov8n_11-21.pt",
            cache_dir="model_cache"
        )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Downloaded model not found at {model_path}")
        
        # Load model with task specification to avoid dataset validation
        model = YOLO(model_path, task='detect')
        
        # Force model to evaluation/inference mode
        model.model.eval()
        
        # Verify model loaded correctly
        if not hasattr(model, 'names'):
            raise ValueError("Model loaded but missing class names")
            
        # Print the existing class names for verification
        st.sidebar.success("✅ Model loaded successfully!")
        st.sidebar.write(f"Detected Classes: {list(model.names.values())}")
        
        # Optional: Run a test inference to verify everything works
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            _ = model(dummy_input, verbose=False)
        except Exception as e:
            raise RuntimeError(f"Model test inference failed: {str(e)}")
        
        return model
        
    except Exception as e:
        st.error("❌ Error loading model:")
        st.error(str(e))
        st.error("\nDetailed error information:")
        st.error(f"- Error type: {type(e).__name__}")
        st.error(f"- Error location: Model loading")
        st.error(f"- Model path: {model_path if 'model_path' in locals() else 'Not created'}")
        return None

def process_image(image, model, conf_threshold=0.25, target_size=(1280, 1280)):
    """Enhanced image processing optimized for high-resolution inputs"""
    try:
        # Get debug state from session state
        show_debug = st.session_state.get('show_debug', False)
        
        # Convert PIL Image to RGB numpy array
        if isinstance(image, Image.Image):
            # Debug original size
            if show_debug:
                st.sidebar.write(f"Original Image Size: {image.size}")
            
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Store original size for scaling
            original_size = image.size
            
            # Preserve aspect ratio while resizing
            ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
            new_size = tuple(int(dim * ratio) for dim in original_size)
            
            # High quality resize
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            if show_debug:
                st.sidebar.write(f"Processing Size: {new_size}")
            
            # Convert to numpy array
            image_np = np.array(image)
        else:
            raise ValueError("Input must be a PIL Image")

        # Run inference
        results = model(image_np, conf=conf_threshold, verbose=False)[0]
        detections = []
        
        # Process detections
        for box in results.boxes:
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
                
            # Draw rectangle with thicker lines for high-res
            thickness = max(2, int(min(image_np.shape[:2]) / 400))
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, thickness)
            
            # Scaled font size for high-res
            font_scale = max(0.5, min(image_np.shape[:2]) / 1000)
            
            # Add label
            label = f"{class_name}: {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(image_np, (x1, y1-label_h-baseline-5), 
                         (x1+label_w, y1), color, -1)
            cv2.putText(image_np, label, (x1, y1-baseline-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 
                       thickness)
            
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
    
    # Initialize session state for debug toggle
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False
    
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
        
        # Image quality settings optimized for high-res
        st.header("Image Settings")
        image_quality = st.select_slider(
            "Image Quality",
            options=["Standard", "High", "Ultra"],
            value="High",
            help="Higher quality may affect performance"
        )
        
        # Quality settings optimized for high-res cameras
        quality_sizes = {
            "Standard": (1024, 1024),    # Base size
            "High": (1280, 1280),        # Recommended
            "Ultra": (1536, 1536)        # Maximum size
        }
        
        # Debug toggle with unique key
        st.session_state.show_debug = st.checkbox(
            "Show Debug Info",
            value=st.session_state.show_debug,
            key="debug_toggle"
        )

    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Camera Input")
        # Add custom HTML/JavaScript for high-res camera
        camera_html = """
        <style>
        .camera-input > label > div > video {
            width: 100% !important;
            height: auto !important;
            max-height: 800px !important;
        }
        .camera-input > label > div {
            width: 100% !important;
            max-width: none !important;
        }
        </style>
        """
        st.markdown(camera_html, unsafe_allow_html=True)
        
        # Camera input with higher resolution
        camera_image = st.camera_input(
            "Take a picture",
            help="Click to capture an image",
            key="high_res_camera",
            on_change=None  # Reset any cached images
        )
        
        # Add file uploader as alternative
        st.write("Or upload an image:")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            key="image_uploader"
        )
        
    with col2:
        st.subheader("Detection Results")
        # Process either camera image or uploaded file
        input_image = camera_image if camera_image is not None else uploaded_file
        
        if input_image is not None:
            try:
                # Process image
                start_time = time.time()
                
                image = Image.open(input_image)
                
                # Show original image size in debug
                if st.session_state.show_debug:
                    st.sidebar.write("Input Image Details:")
                    st.sidebar.write(f"Original Size: {image.size}")
                    st.sidebar.write(f"Mode: {image.mode}")
                    if hasattr(input_image, 'size'):
                        st.sidebar.write(f"File Size: {input_image.size / 1024:.1f} KB")
                
                target_size = quality_sizes[image_quality]
                
                # Process image
                annotated_img, detections = process_image(
                    image, 
                    model, 
                    conf_threshold,
                    target_size=target_size
                )
                
                if annotated_img is not None:
                    # Display results
                    st.image(annotated_img, channels="RGB", use_column_width=True)
                    
                    if detections:
                        st.write("Detections:")
                        for det in detections:
                            conf = det['confidence']
                            color = 'green' if conf > 0.8 else 'orange' if conf > 0.5 else 'red'
                            st.markdown(f"- {det['class']}: ::{color}[{conf:.2f}]")
                    else:
                        st.info("No cats detected in image")
                    
                    # Show processing time if debug is enabled
                    if st.session_state.show_debug:
                        process_time = time.time() - start_time
                        st.sidebar.write(f"Processing Time: {process_time:.3f} seconds")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.write("Please try taking another picture.")
                if st.session_state.show_debug:
                    st.error(f"Detailed error: {str(e)}")
    
    # Add information about supported resolutions
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Supported Image Sizes")
        for quality, size in quality_sizes.items():
            st.write(f"- {quality}: {size[0]}x{size[1]}")
        
        if st.session_state.show_debug:
            st.markdown("---")
            st.markdown("### System Information")
            st.write(f"- CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                st.write(f"- GPU: {torch.cuda.get_device_name(0)}")

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