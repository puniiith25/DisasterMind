import json

INPUT = "data/disasterm3/disasterm3_qa.json"
OUTPUT = "data/disasterm3/disasterm3_qa.json"

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

original_count = len(data)

# Remove records with empty answers
cleaned = [
    item
    for item in data
    if item.get("answer", "").strip()
]

removed = original_count - len(cleaned)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(
        cleaned,
        f,
        indent=2,
        ensure_ascii=False
    )

print("Original records:", original_count)
print("Removed invalid:", removed)
print("Final records:", len(cleaned))