import json
from pathlib import Path

INPUT = Path("data/disasterm3/selected_samples.json")

OUTPUT = Path(
    "data/disasterm3/disasterm3_qa.json"
)

IMAGE_ROOT = Path(
    "data/disasterm3/raw/DisasterM3_Bench"
)

with open(INPUT, "r", encoding="utf-8") as f:
    samples = json.load(f)

final_dataset = []

for index, item in enumerate(samples):

    record = {
        "id": f"dm3_{index:05d}",
        "task": item["task"],
        "question": item["prompts"],
        "answer": str(item["ground_truth"]),
        "image_type": item.get(
            "post_image_type",
            item.get("image_type")
        ),
        "pre_image": None,
        "post_image": None,
        "image": None,
        "options": item.get("options_list"),
        "answer_option": item.get(
            "ground_truth_option"
        ),
        "description": item.get(
            "cls_description"
        )
    }

    # Pre-disaster image
    if "pre_image_path" in item:

        relative = item[
            "pre_image_path"
        ].replace("\\", "/")

        record["pre_image"] = str(
            IMAGE_ROOT / relative
        )

    # Post-disaster image
    if "post_image_path" in item:

        relative = item[
            "post_image_path"
        ].replace("\\", "/")

        record["post_image"] = str(
            IMAGE_ROOT / relative
        )

    # Single image
    if "image_path" in item:

        relative = item[
            "image_path"
        ].replace("\\", "/")

        record["image"] = str(
            IMAGE_ROOT / relative
        )

    final_dataset.append(record)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(
        final_dataset,
        f,
        indent=2,
        ensure_ascii=False
    )

print("Final dataset created")
print("Total records:", len(final_dataset))
print("Saved:", OUTPUT)

print("\nExample:")
print(
    json.dumps(
        final_dataset[0],
        indent=2,
        ensure_ascii=False
    )
)