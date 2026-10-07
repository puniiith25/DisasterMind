import json
import random
from collections import defaultdict

INPUT = "data/disasterm3/DisasterM3_Bench/benchmark_release.json"
OUTPUT = "data/disasterm3/selected_samples.json"

# Number of samples we initially want from each task
SAMPLES_PER_TASK = {
    "Disaster Bearing Bodies Recognition": 300,
    "Building Damage Counting": 300,
    "Road Damage Counting": 300,
    "Disaster Scene Recognition": 300,
    "Disaster Type Recognition": 200,
    "Relational Reasoning": 200,
}

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

# Group records by task
grouped = defaultdict(list)

for item in data:
    task = item["task"]

    if task in SAMPLES_PER_TASK:
        grouped[task].append(item)

# Fixed seed = reproducible experiment
random.seed(42)

selected = []

for task, number in SAMPLES_PER_TASK.items():

    available = grouped[task]

    if len(available) < number:
        print(
            f"WARNING: {task} only has "
            f"{len(available)} samples"
        )
        chosen = available
    else:
        chosen = random.sample(available, number)

    selected.extend(chosen)

    print(
        f"{task}: selected {len(chosen)} "
        f"from {len(available)}"
    )

# Save selected samples
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(
        selected,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\nTotal selected:", len(selected))
print("Saved to:", OUTPUT)