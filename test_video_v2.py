
from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/pothole/second_training/weights/best.pt"
)

# Replace with a video containing road footage
model.predict(
    source="road.mp4",
    device="mps",
    conf=0.20,
    save=True,
    stream=True,
    project="runs/pothole",
    name="video_test_v2"
)

# Process all frames (required with stream=True)
for result in model.predict(
    source="road.mp4",
    device="mps",
    conf=0.20,
    save=True,
    stream=True,
    project="runs/pothole",
    name="video_test_v2"
):
    pass

print("Video analysis completed!")
