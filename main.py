import cv2
from ultralytics import YOLO

def main():
    # 1. Load the YOLO model
    model = YOLO("yolov8m.pt")

    # 2. Open the video file
    video_path = "test_video.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'. Make sure the file exists.")
        return

    print("Starting video processing... Press 'q' in the video window to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Reached the end of the video or video stream disconnected.")
            break

        # 3. Run YOLO inference on the current frame
        results = model(frame, imgsz=640, conf=0.3)

        # Count what is currently on the screen
        current_in_frame = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "person": 0}

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for cls_tensor in boxes.cls:
                class_name = model.names[int(cls_tensor)]
                if class_name in current_in_frame:
                    current_in_frame[class_name] += 1

        # 4. Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Draw live counts - one per line, bigger text
        y_pos = 30
        for vehicle_type, count in current_in_frame.items():
            text = f"{vehicle_type.capitalize()}s: {count}"
            cv2.putText(annotated_frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_pos += 35

        # 5. Display the frame
        cv2.imshow("Traffic Analytics - YOLO Inference", annotated_frame)

        # Check if user clicked 'X'
        if cv2.getWindowProperty("Traffic Analytics - YOLO Inference", cv2.WND_PROP_VISIBLE) < 1:
            break

        # Check for 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
