from ultralytics import solutions

def main():
    try:
        # Use basic call without additional parameters
        solutions.inference(
            model="cats_yolov8n_11-21-v2.pt"
        )
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()