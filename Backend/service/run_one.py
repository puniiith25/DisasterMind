import json

from service.qwen_vlm import answer_question


DATASET = "data/disasterm3/disasterm3_qa.json"


# Load dataset
with open(DATASET, "r", encoding="utf-8") as f:
    data = json.load(f)


# Take first sample
sample = data[0]

print("=" * 70)
print("DISASTER M3 - QWEN BASELINE TEST")
print("=" * 70)

print("ID:", sample["id"])
print("Task:", sample["task"])

print("\nQuestion:")
print(sample["question"])

print("\nGround Truth:")
print(sample["answer"])

print("\nPre image:")
print(sample["pre_image"])

print("\nPost image:")
print(sample["post_image"])

print("\nRunning Qwen2.5-VL...")
print("-" * 70)


prediction = answer_question(
    sample["pre_image"],
    sample["post_image"],
    sample["question"]
)

print("\nQWEN PREDICTION:")
print(prediction)

print("\n" + "=" * 70)