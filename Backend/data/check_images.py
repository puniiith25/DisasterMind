import json
from pathlib import Path

INPUT = "data/disasterm3/selected_samples.json"

with open(INPUT, "r", encoding="utf-8") as f:
    samples = json.load(f)

pre_images = set()
post_images = set()
single_images = set()

for item in samples:

    if "pre_image_path" in item:
        pre_images.add(
            item["pre_image_path"].replace("\\", "/")
        )

    if "post_image_path" in item:
        post_images.add(
            item["post_image_path"].replace("\\", "/")
        )

    if "image_path" in item:
        single_images.add(
            item["image_path"].replace("\\", "/")
        )

print("Selected samples:", len(samples))

print("\nUnique pre-disaster images:", len(pre_images))
print("Unique post-disaster images:", len(post_images))
print("Unique single images:", len(single_images))

all_images = pre_images | post_images | single_images

print("Total unique image paths:", len(all_images))

print("\nFirst 20 image paths:")

for image in sorted(all_images)[:20]:
    print(image)