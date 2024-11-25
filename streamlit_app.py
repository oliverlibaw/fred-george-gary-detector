from ultralytics import YOLO
from ultralytics import solutions
import streamlit as st
from huggingface_hub import hf_hub_download


st.set_page_config(page_title="Cat Detection App", page_icon="🐱")

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="oliverlibaw/fred-george-gary-11-2024.pt",
        filename="cats_yolov8n_11-21-v2.pt",
        cache_dir="model_cache"
    )
    return YOLO(model_path)


def main():
    model_path = hf_hub_download(
        repo_id="oliverlibaw/fred-george-gary-11-2024.pt",
        filename="cats_yolov8n_11-21-v2.pt",
        cache_dir="model_cache"
    )
    solutions.inference(model_path)

if __name__ == "__main__":
    main()