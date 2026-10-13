from service.embeddings import embed_text


text = """
A tornado damaged buildings, roads and forest
in the affected area.
"""

embedding = embed_text(text)

print("Embedding generated")
print("Dimensions:", len(embedding))
print("First 10 values:")
print(embedding[:10])