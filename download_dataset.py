
from huggingface_hub import HfApi, hf_hub_download

repo = "dronefreak/HRP4K"
api = HfApi()

files = api.list_repo_files(repo, repo_type="dataset")
file_set = set(files)

images = sorted(
    f for f in files
    if f.startswith("data/images/train/")
    and f.lower().endswith((".jpg", ".jpeg", ".png"))
)

downloaded = 0

for image in images:
    label = image.replace(
        "data/images/train/",
        "data/labels/train/",
        1
    ).rsplit(".", 1)[0] + ".txt"

    if label not in file_set:
        continue

    for filename in (image, label):
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=filename,
            local_dir="datasets/potholes"
        )

    downloaded += 1
    print(f"Downloaded {downloaded}/500")

    if downloaded == 500:
        break

print(f"\nCompleted: {downloaded} image-label pairs")
