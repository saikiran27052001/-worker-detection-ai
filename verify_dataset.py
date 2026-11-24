
from pathlib import Path

base = Path("datasets/potholes/ready")

for split in ["train", "val"]:
    images = sorted((base / "images" / split).glob("*.jpg"))

    total_boxes = 0
    invalid_boxes = 0
    missing_labels = 0

    for image in images:
        label = base / "labels" / split / f"{image.stem}.txt"

        if not label.exists():
            missing_labels += 1
            continue

        for line in label.read_text().splitlines():
            if not line.strip():
                continue

            values = list(map(float, line.split()))

            if len(values) != 5:
                invalid_boxes += 1
                continue

            cls, x, y, w, h = values
            total_boxes += 1

            if (
                cls != 0 or
                w <= 0 or h <= 0 or
                x - w / 2 < 0 or
                x + w / 2 > 1 or
                y - h / 2 < 0 or
                y + h / 2 > 1
            ):
                invalid_boxes += 1

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Annotated objects:", total_boxes)
    print("Missing labels:", missing_labels)
    print("Invalid boxes:", invalid_boxes)
