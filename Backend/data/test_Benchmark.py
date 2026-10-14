import json 
path="data/disasterm3/DisasterM3_Bench/benchmark_release.json"


with open(path,'r',encoding='utf-8') as f:
    data = json.load(f)
print("Type:", type(data))

if isinstance(data ,dict):
    print("Keys:")
    for key in data.keys:
        print(" _",key)

elif isinstance(data, list):

    print("Number of records:", len(data))

print("\nFirst part of dataset:\n")

print(

    json.dumps(

        data,

        indent=2,

        ensure_ascii=False

    )[:10000]

)