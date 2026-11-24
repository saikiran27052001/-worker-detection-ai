from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="datasets/potholes/ready_v2/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="mps",
    workers=0,
    project="runs/pothole",
    name="second_training"
)

print("Second training completed!")
