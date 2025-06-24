from ultralytics import YOLO
import cv2
import os
import random

# Load your trained YOLO model
model = YOLO("../runs_extracted/detect/train/weights/best.pt")  

# Define input and output paths
image_path = "test_sources/tomatoes_test.jpg"
output_folder = "test_outputs"
output_image_path = os.path.join(output_folder, "output.jpg")

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Read the image
image = cv2.imread(image_path)

# Run YOLO detection
results = model(image)

# Assign a random color to each class
num_classes = len(model.names)
class_colors = {i: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(num_classes)}

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
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save the output image
cv2.imwrite(output_image_path, image)

cv2.imshow("YOLO Image Detection", image)
cv2.waitKey(0)  
cv2.destroyAllWindows()