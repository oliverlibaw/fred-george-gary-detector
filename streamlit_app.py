from ultralytics import YOLO
import streamlit as st
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image

st.set_page_config(page_title="Cat Detection App", page_icon="🐱")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="oliverlibaw/fred-george-gary-11-2024.pt",
        filename="cats_yolov8n_11-21-v2.pt",
        cache_dir="model_cache"
    )
    return YOLO(model_path)

def process_image(image, model, conf):
    results = model.predict(np.array(image), conf=conf)[0]
    return Image.fromarray(results.plot())

def main():
    st.title("Cat Detection App")
    model = load_model()
    
    conf = st.sidebar.slider("Confidence", 0.0, 1.0, 0.25)
    
    img_source = st.camera_input("Take a picture")
    if not img_source:
        img_source = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png"])
        
    if img_source:
        input_image = Image.open(img_source)
        result_image = process_image(input_image, model, conf)
        st.image(result_image)

if __name__ == "__main__":
    main()