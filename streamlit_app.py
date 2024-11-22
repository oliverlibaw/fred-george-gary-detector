def main():
    st.title("Real-time Object Detection with Sound")
    
    # Initialize audio system
    audio_initialized = initialize_audio()
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please check your Hugging Face repository settings.")
        st.stop()

    # Add confidence threshold slider in sidebar
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )
    
    # Add image quality settings
    st.sidebar.header("Camera Settings")
    image_quality = st.sidebar.select_slider(
        "Image Quality",
        options=["Low", "Medium", "High"],
        value="High"
    )
    
    # Map quality settings to image sizes
    quality_sizes = {
        "Low": (320, 320),
        "Medium": (640, 640),
        "High": (1280, 1280)
    }

    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Camera Input")
        camera_image = st.camera_input("Take a picture")
        
    with col2:
        st.write("Detection Results")
        if camera_image is not None:
            try:
                # Add image quality checks
                image = Image.open(camera_image)
                
                # Check image size
                if image.size[0] < 320 or image.size[1] < 320:
                    st.warning("Image resolution is very low. This may affect detection quality.")
                
                # Check image mode and convert if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Process image
                annotated_img, detections = process_image(image, model, conf_threshold)
                
                # Display results
                st.image(annotated_img, channels="RGB", use_column_width=True)
                
                # Display detection information with confidence colors
                st.write("Detections:")
                for det in detections:
                    conf = det['confidence']
                    color = 'green' if conf > 0.8 else 'orange' if conf > 0.5 else 'red'
                    st.markdown(f"- {det['class']}: ::{color}[{conf:.2f}]")
                
                # Show debug information if requested
                if st.sidebar.checkbox("Show Debug Info"):
                    st.write("Image Information:")
                    st.write(f"Size: {image.size}")
                    st.write(f"Mode: {image.mode}")
                    st.write(f"Format: {image.format}")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.write("Please try taking another picture.")