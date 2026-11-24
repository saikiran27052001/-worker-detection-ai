from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo26n.pt")

# Fine-tune for pothole detection
results = model.train(
    data="datasets/potholes/ready/data.yaml",
    epochs=20,
    imgsz=640,
    batch=8,
    device="mps",
    workers=0,
    project="runs/pothole",
    name="first_training"
)

print("Training completed!")

