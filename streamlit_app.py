from ultralytics import solutions
import streamlit as st

# Must be first Streamlit command
st.set_page_config(page_title="Cat Detection App", page_icon="🐱")


def main():
    st.title("Cat Detection")
    
    with st.sidebar:
        confidence = st.slider("Confidence", 0.0, 1.0, 0.25)
        iou_threshold = st.slider("IoU", 0.0, 1.0, 0.7)
    
    try:
        solutions.inference(
            model="cats_yolov8n_11-21-v2.pt"
        )
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()