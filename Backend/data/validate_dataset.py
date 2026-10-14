import json
from pathlib import Path

DATASET = Path("data/disasterm3/disasterm3_qa.json")

with open(DATASET, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total records:", len(data))

errors = []

for item in data:

    if not item.get("question"):
        errors.append((item["id"], "missing question"))

    if not item.get("answer"):
        errors.append((item["id"], "missing answer"))

    # Check available image
    image_paths = [
        item.get("image"),
        item.get("pre_image"),
        item.get("post_image")
    ]

    valid_image = False

    for path in image_paths:
        if path and Path(path).exists():
            valid_image = True
            break

    if not valid_image:
        errors.append((item["id"], "missing image"))

print("Errors:", len(errors))

if errors:
    print("\nFirst errors:")
    for error in errors[:20]:
        print(error)
else:
    print("\nDataset validation successful.")