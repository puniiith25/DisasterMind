import json
from pathlib import Path

INPUT = Path("data/disasterm3/selected_samples.json")
IMAGE_ROOT = Path("data/disasterm3/raw/DisasterM3_Bench")

with open(INPUT, "r", encoding="utf-8") as f:
    samples = json.load(f)

image_paths = set()

for item in samples:

    for key in [
        "pre_image_path",
        "post_image_path",
        "image_path"
    ]:

        if key in item:
            path = item[key].replace("\\", "/")
            image_paths.add(path)

missing = []
found = []

for relative_path in sorted(image_paths):

    file_path = IMAGE_ROOT / relative_path

    if file_path.exists():
        found.append(relative_path)
    else:
        missing.append(relative_path)

print("Total required images:", len(image_paths))
print("Images found:", len(found))
print("Images missing:", len(missing))

if missing:
    print("\nMissing images:")
    for path in missing[:20]:
        print(path)
else:
    print("\nAll required images are available.")