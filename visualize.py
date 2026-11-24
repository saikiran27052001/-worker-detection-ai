
import cv2
from pathlib import Path

# Select the first downloaded image
folder = Path("datasets/potholes/data")
image_path = sorted((folder / "images/train").glob("*.jpg"))[25]

# Find its matching annotation file
label_path = folder / "labels/train" / (image_path.stem + ".txt")

image = cv2.imread(str(image_path))
height, width = image.shape[:2]

print("Image:", image_path.name)
print("Image size:", width, "x", height)

# Read the YOLO annotations
with open(label_path) as file:
    for line in file:
        values = list(map(float, line.split()))
        class_id, x, y, w, h = values[:5]

        # Convert normalized coordinates to pixels
        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)

        # Draw a green bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2),
                      (0, 255, 0), 2)

        print("Object class:", int(class_id))

# Save the annotated image
cv2.imwrite("annotated_pothole.jpg", image)

print("Saved annotated_pothole.jpg")
