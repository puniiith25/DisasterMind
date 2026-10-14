import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from service.clip_embeddings import embed_image


CORPUS_FILE = Path(
    "data/disasterm3/retrieval/image_corpus.json"
)

QDRANT_PATH = "data/qdrant"

COLLECTION_NAME = "disasterm3_images"

VECTOR_SIZE = 512


def main():

    # Load image corpus
    with open(
        CORPUS_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        corpus = json.load(f)

    print(f"Loaded {len(corpus)} images")

    # Open local Qdrant
    client = QdrantClient(
        path=QDRANT_PATH
    )

    # Remove old collection if it exists
    if client.collection_exists(
        collection_name=COLLECTION_NAME
    ):
        print(
            f"Deleting existing collection: "
            f"{COLLECTION_NAME}"
        )

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

    # Create image collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    points = []

    for index, item in enumerate(corpus):

        image_path = item["image_path"]

        print(
            f"Embedding image "
            f"{index + 1}/{len(corpus)}"
        )

        try:

            vector = embed_image(
                image_path
            )

            points.append(
                PointStruct(
                    id=index,
                    vector=vector,
                    payload=item,
                )
            )

        except Exception as e:

            print(
                f"\nERROR processing:"
            )
            print(image_path)
            print(e)

    print()
    print(
        f"Uploading {len(points)} image vectors..."
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print()
    print("=" * 60)
    print("IMAGE QDRANT INDEXING COMPLETED")
    print("=" * 60)
    print(
        f"Collection: {COLLECTION_NAME}"
    )
    print(
        f"Vectors: {len(points)}"
    )


if __name__ == "__main__":
    main()