import cv2
import os

video = cv2.VideoCapture("traffic.mp4")

os.makedirs("frames", exist_ok=True)

frame_number = 0
saved = 0

fps = video.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    raise ValueError("Could not read video FPS")

while True:
    success, frame = video.read()

    if not success:
        break

    # Save one frame every second
    if frame_number % round(fps) == 0:
        cv2.imwrite(f"frames/frame_{saved}.jpg", frame)
        saved += 1

    frame_number += 1

video.release()

print(f"Extracted {saved} images!")
