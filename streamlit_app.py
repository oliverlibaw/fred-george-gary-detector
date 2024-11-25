import streamlit as st
from ultralytics import solutions
from huggingface_hub import hf_hub_download

# Set page config
st.set_page_config(
    page_title="Cat Detection App",
    page_icon="🐱",
    layout="wide"
)

@st.cache_resource
def get_model_path():
    try:
        return hf_hub_download(
            repo_id="oliverlibaw/fred-george-gary-11-2024.pt",
            filename="cats_yolov8n_11-21-v2.pt",
            cache_dir="model_cache",
            force_download=True
        )
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return None

def main():
    st.title("Cat Detection App")
    
    # Get model path
    model_path = get_model_path()
    if model_path is None:
        st.error("Failed to load model. Please refresh the page.")
        st.stop()
    
    # Settings sidebar
    with st.sidebar:
        st.header("Detection Settings")
        conf_threshold = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05
        )

    # Main content area
    st.camera_input("Take a picture")
    st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])

    # Run inference
    solutions.inference(
        source=None,  # Will use camera/upload input
        model=model_path,
        conf=conf_threshold
    )

if __name__ == '__main__':
    main()
    