from qdrant_client import QdrantClient

from service.clip_embeddings import embed_image


QDRANT_PATH = "data/qdrant"
COLLECTION_NAME = "disasterm3_images"


def retrieve_images(
    image_path: str,
    top_k: int = 5,
):

    client = QdrantClient(
        path=QDRANT_PATH
    )

    query_vector = embed_image(
        image_path
    )

    # Retrieve extra results because
    # some results will be filtered out.
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k + 10,
        with_payload=True,
    )

    filtered_results = []

    for result in results.points:

        payload = result.payload

        retrieved_path = payload.get(
            "image_path"
        )

        # Remove exact query image
        if retrieved_path == image_path:
            continue

        filtered_results.append(result)

        if len(filtered_results) >= top_k:
            break

    return filtered_results


if __name__ == "__main__":

    query_image = (
        "data/disasterm3/raw/"
        "DisasterM3_Bench/test_images/"
        "joplin_tornado_00000065_post_disaster.png"
    )

    results = retrieve_images(
        query_image,
        top_k=5
    )

    print()
    print("QUERY IMAGE")
    print("=" * 60)
    print(query_image)

    print()
    print("SIMILAR HISTORICAL IMAGES")
    print("=" * 60)

    for i, result in enumerate(
        results,
        start=1
    ):

        payload = result.payload

        print(
            f"\n[{i}] Score: "
            f"{result.score:.4f}"
        )

        print(
            "Image:",
            payload.get("image_path")
        )

        print(
            "Task:",
            payload.get("task")
        )

        print(
            "Source:",
            payload.get("source")
        )