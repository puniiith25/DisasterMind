import json

path = "data/disasterm3/selected_samples.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

for index, item in enumerate(data):
    if index == 274:
        print(json.dumps(item, indent=2, ensure_ascii=False))
        break