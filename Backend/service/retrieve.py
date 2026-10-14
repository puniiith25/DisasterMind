from qdrant_client import QdrantClient

from service.embeddings import embed_text


QDRANT_PATH = "data/qdrant"
COLLECTION_NAME = "disasterm3_historical"


def retrieve(
    question: str,
    task: str | None = None,
    top_k: int = 5
):

    client = QdrantClient(
        path=QDRANT_PATH
    )

    if task:
        query_text = f"""
Task: {task}

Question:
{question}
""".strip()
    else:
        query_text = question

    query_vector = embed_text(query_text)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    )

    return results.points


if __name__ == "__main__":

    task = "Disaster Bearing Bodies Recognition"

    question = (
        "What categories of objects have "
        "sustained damage in the affected area?"
    )

    results = retrieve(
        question=question,
        task=task,
        top_k=5
    )

    print("\nQUERY")
    print("=" * 60)
    print(f"Task: {task}")
    print(f"Question: {question}")

    print("\nRETRIEVED HISTORICAL CONTEXT")
    print("=" * 60)

    for i, result in enumerate(results, start=1):

        payload = result.payload

        print(f"\n[{i}] Score: {result.score:.4f}")
        print(f"Task: {payload.get('task')}")
        print(f"Question: {payload.get('question')}")
        print(f"Historical answer: {payload.get('answer')}")
        print(f"Source: {payload.get('source')}")