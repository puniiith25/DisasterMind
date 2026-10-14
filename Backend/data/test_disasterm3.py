from datasets import load_dataset

DATASET = "Kingdrone-Junjue/DisasterM3"

print("Loading Dataset...")
dataset = load_dataset(
    DATASET,
    split='test'
                       )

print("\n Dataset")
print(dataset)
print(dataset.column_names)
print(dataset[0])