
from pathlib import Path
import random
import shutil

source = Path("datasets/potholes/data")
destination = Path("datasets/potholes/ready_v2")

# Find images with matching labels
images = sorted((source / "images/train").glob("*.jpg"))

pairs = [
    image for image in images
    if (source / "labels/train" / f"{image.stem}.txt").exists()
]

print("Available image-label pairs:", len(pairs))

# Shuffle using a fixed seed
random.seed(42)
random.shuffle(pairs)

# Split: 80% train, 10% validation, 10% test
n = len(pairs)
train_images = pairs[:int(n * 0.8)]
val_images = pairs[int(n * 0.8):int(n * 0.9)]
test_images = pairs[int(n * 0.9):]

# Create a fresh dataset folder
if destination.exists():
    shutil.rmtree(destination)

for split_name, image_list in [
    ("train", train_images),
    ("val", val_images),
    ("test", test_images)
]:
    for image in image_list:
        label = source / "labels/train" / f"{image.stem}.txt"

        image_dir = destination / "images" / split_name
        label_dir = destination / "labels" / split_name

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(image, image_dir / image.name)
        shutil.copy2(label, label_dir / label.name)

# Create YOLO configuration
yaml_content = f"""
path: {destination.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: pothole
"""

(destination / "data.yaml").write_text(yaml_content)

print("Training:", len(train_images))
print("Validation:", len(val_images))
print("Testing:", len(test_images))
print("Dataset ready!")
