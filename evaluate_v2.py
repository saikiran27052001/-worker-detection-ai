from ultralytics import YOLO

# Load the newly trained model
model = YOLO(
    "runs/detect/runs/pothole/second_training/weights/best.pt"
)

# Evaluate on the separate test dataset
metrics = model.val(
    data="datasets/potholes/ready_v2/data.yaml",
    split="test",
    device="mps",
    plots=True,
    project="runs/pothole",
    name="evaluation_v2"
)

print("\n--- TEST RESULTS ---")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")
print(f"mAP@50: {metrics.box.map50:.3f}")
print(f"mAP@50-95: {metrics.box.map:.3f}")
