from ultralytics import YOLO
import cv2
import os
import random

# Load your trained YOLO model
model = YOLO("../runs_extracted/detect/train/weights/best.pt")  

# Define output folder and file
output_folder = "test_outputs"
output_video_path = os.path.join(output_folder, "webcam_output.mp4")

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Get webcam properties
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Assign a random color to each class
num_classes = len(model.names)
class_colors = {i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(num_classes)}

# Process the webcam feed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if no frame is captured

    # Flip the frame horizontally for the mirror effect
    frame = cv2.flip(frame, 1)

    # Run YOLO detection on the frame
    results = model(frame)

    # Draw bounding boxes and labels
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integer coordinates
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class ID
            label = f"{model.names[cls]} {conf:.2f}"  # Class name + confidence

            # Get color for the class
            color = class_colors.get(cls, (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the mirrored frame
    cv2.imshow("YOLO Webcam Detection", frame)

    # Write the processed frame to the output video
    out.write(frame)

    if cv2.waitKey(1) == 27 or cv2.getWindowProperty("YOLO Webcam Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()