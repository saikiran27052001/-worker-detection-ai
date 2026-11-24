from ultralytics import YOLO

# Load your trained pothole model
model = YOLO(
    "runs/detect/runs/pothole/first_training/weights/best.pt"
)

# Test it on a validation image
results = model.predict(
    source="datasets/potholes/ready/images/val",
    device="mps",
    conf=0.12,
    save=True
)

print("Testing completed!")
