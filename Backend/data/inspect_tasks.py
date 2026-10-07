import json
from collections import defaultdict

path = "data/disasterm3/DisasterM3_Bench/benchmark_release.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = {}

for item in data:
    task = item["task"]

    if task not in samples:
        samples[task] = item

for task, item in samples.items():
    print("\n" + "=" * 80)
    print("TASK:", task)
    print("=" * 80)

    for key, value in item.items():
        print(f"{key}: {value}")