import json
from pathlib import Path
from huggingface_hub import snapshot_download

# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT = Path("data/disasterm3/selected_samples.json")
OUTPUT_DIR = Path("data/disasterm3/raw")

REPO_ID = "Kingdrone-Junjue/DisasterM3"

# --------------------------------------------------
# Load selected samples
# --------------------------------------------------

with open(INPUT, "r", encoding="utf-8") as f:
    samples = json.load(f)

# --------------------------------------------------
# Collect unique image paths
# --------------------------------------------------

image_paths = set()

for item in samples:

    if "pre_image_path" in item:
        image_paths.add(
            item["pre_image_path"].replace("\\", "/")
        )

    if "post_image_path" in item:
        image_paths.add(
            item["post_image_path"].replace("\\", "/")
        )

    if "image_path" in item:
        image_paths.add(
            item["image_path"].replace("\\", "/")
        )

print("Selected samples:", len(samples))
print("Unique images:", len(image_paths))

# --------------------------------------------------
# Convert paths to actual Hugging Face paths
# --------------------------------------------------

patterns = [
    f"DisasterM3_Bench/{path}"
    for path in sorted(image_paths)
]

print("\nDownloading only required images...")
print("Number of files:", len(patterns))

# --------------------------------------------------
# Download
# --------------------------------------------------

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=str(OUTPUT_DIR),
    allow_patterns=patterns,
    max_workers=8
)

print("\nDownload completed.")
print("Images stored in:", OUTPUT_DIR)