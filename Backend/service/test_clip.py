from service.clip_embeddings import embed_image


IMAGE = (
    "data/disasterm3/raw/"
    "DisasterM3_Bench/test_images/"
    "joplin_tornado_00000065_post_disaster.png"
)


embedding = embed_image(IMAGE)

print()
print("CLIP embedding generated")
print("Dimensions:", len(embedding))
print("First 10 values:")
print(embedding[:10])