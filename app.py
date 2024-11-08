import streamlit as st
import os
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(page_title="Object Detection App", layout="wide")

def load_dependencies():
    """Load heavy dependencies with error handling"""
    try:
        from ultralytics import YOLO
        import torch
        return True
    except ImportError as e:
        st.error(f"Error loading dependencies: {str(e)}")
        st.info("If you're seeing this in development, try deploying to Streamlit Cloud where all dependencies will be properly installed.")
        return False

def main():
    st.title("Object Detection with YOLOv8")
    st.write("Upload an image to detect objects")
    
    # Only try to load model if dependencies are available
    if not load_dependencies():
        st.stop()
    
    # Now it's safe to import YOLO
    from ultralytics import YOLO
    
    # Load model with error handling
    @st.cache_resource
    def load_model():
        try:
            model_path = os.path.join('models', 'best.pt')
            return YOLO(model_path)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        st.stop()
    
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
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
                                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == '__main__':
    main()