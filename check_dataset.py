from pathlib import Path
from collections import Counter

folder = Path("datasets/potholes/data")
labels = list((folder / "labels/train").glob("*.txt"))

classes = Counter()

for label in labels:
    for line in label.read_text().splitlines():
        if line.strip():
            class_id = int(line.split()[0])
            classes[class_id] += 1

print("Label files:", len(labels))
print("Classes found:", dict(classes))

yaml_path = folder / "data.yaml"

if yaml_path.exists():
    print("\nDataset configuration:")
    print(yaml_path.read_text())
else:
    print("\nNo data.yaml found in data folder.")
