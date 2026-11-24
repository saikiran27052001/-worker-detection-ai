
from ultralytics import YOLO

# Load our pretrained model
model = YOLO("yolo26n.pt")

# Process the video frame by frame
results = model.predict(
    source="traffic.mp4",
    device="mps",
    save=True,
    conf=0.4,
    stream=True
)

# Print detections for each frame
for frame_number, result in enumerate(results, start=1):
    detections = []

    for box in result.boxes:
        name = result.names[int(box.cls[0])]
        detections.append(name)

    print(f"Frame {frame_number}: {detections}")

print("Video processing completed!")
