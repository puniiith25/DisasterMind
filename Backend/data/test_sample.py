import json
from pathlib import Path

DATASET = Path(
    "data/disasterm3/disasterm3_qa.json"
)

with open(DATASET, "r", encoding="utf-8") as f:
    data = json.load(f)

sample = data[0]

print("ID:", sample["id"])
print("Task:", sample["task"])
print("Question:", sample["question"])
print("Ground truth:", sample["answer"])

print("\nPre image:")
print(sample["pre_image"])

print("\nPost image:")
print(sample["post_image"])