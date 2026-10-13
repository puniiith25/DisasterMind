import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

from service.embeddings import embed_text


CORPUS_FILE = Path(
    "data/disasterm3/retrieval/dev_corpus.json"
)

QDRANT_PATH = "data/qdrant"

COLLECTION_NAME = "disasterm3_historical"

VECTOR_SIZE = 1024


def build_text(item):

    return f"""
Task: {item.get("task", "")}

Question:
{item.get("question", "")}

Historical answer:
{item.get("answer", "")}
""".strip()


def main():

    with open(
        CORPUS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        corpus = json.load(f)

    print(f"Loaded {len(corpus)} records")

    client = QdrantClient(
        path=QDRANT_PATH
    )

    # Remove old collection if it exists
    if client.collection_exists(
        collection_name=COLLECTION_NAME
    ):
        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

    # Create collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

    points = []

    for index, item in enumerate(corpus):

        text = build_text(item)

        print(
            f"Embedding {index + 1}/{len(corpus)}"
        )

        vector = embed_text(text)

        points.append(
            PointStruct(
                id=index,
                vector=vector,
                payload=item
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    print()
    print("Qdrant indexing completed")
    print(
        "Collection:",
        COLLECTION_NAME
    )
    print(
        "Vectors:",
        len(points)
    )


if __name__ == "__main__":
    main()