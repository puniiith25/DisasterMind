import json

path = "data/disasterm3/disasterm3_qa.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if not item.get("answer"):
        print("Problematic record:")
        print(json.dumps(item, indent=2, ensure_ascii=False))