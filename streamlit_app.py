from ultralytics import solutions
import streamlit as st

st.set_page_config(page_title="Cat Detection App", page_icon="🐱")

def main():
   st.title("Cat Detection")
   
   with st.sidebar:
       conf = st.slider("Confidence", 0.0, 1.0, 0.25)
       iou = st.slider("IoU", 0.0, 1.0, 0.45)
   
   try:
       solutions.inference(
           model="cats_yolov8n_11-21-v2.pt",
           conf=conf,
           iou=iou
       )
   except Exception as e:
       st.error(f"Error: {str(e)}")

if __name__ == "__main__":
   main()