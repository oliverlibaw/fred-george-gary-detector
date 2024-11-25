from ultralytics import solutions
import streamlit as st
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Cat Detection App", page_icon="🐱")

@st.cache_resource
def get_model_path():
    return hf_hub_download(
        repo_id="oliverlibaw/fred-george-gary-11-2024.pt",
        filename="cats_yolov8n_11-21-v2.pt"
    )

def main():
    st.title("Cat Detection")
    
    # Sliders in sidebar
    with st.sidebar:
        conf = st.slider("Confidence", 0.0, 1.0, 0.25)
        iou = st.slider("IoU", 0.0, 1.0, 0.45)
    
    try:
        model_path = get_model_path()
        solutions.inference(
            source=None,  # Uses webcam by default
            model=model_path,
            conf=conf,
            iou=iou
        )
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()