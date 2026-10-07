import json
from collections import Counter

path = "data/disasterm3/DisasterM3_Bench/benchmark_release.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total records:", len(data))

tasks = Counter(item["task"] for item in data)

print("\nAvailable Tasks:")
print("-" * 60)

for task, count in tasks.most_common():
    print(f"{task}: {count}")