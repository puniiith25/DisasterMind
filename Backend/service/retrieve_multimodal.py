from service.retrieve import retrieve
from service.retrieve_images import retrieve_images


def retrieve_multimodal(
    image_path: str,
    question: str,
    task: str | None = None,
    text_top_k: int = 5,
    image_top_k: int = 5,
):

    # -------------------------
    # TEXT RETRIEVAL
    # -------------------------

    text_results = retrieve(
        question=question,
        task=task,
        top_k=text_top_k,
    )

    # -------------------------
    # IMAGE RETRIEVAL
    # -------------------------

    image_results = retrieve_images(
        image_path=image_path,
        top_k=image_top_k,
    )

    return {
        "text_results": text_results,
        "image_results": image_results,
    }


if __name__ == "__main__":

    image = (
        "data/disasterm3/raw/"
        "DisasterM3_Bench/test_images/"
        "joplin_tornado_00000065_post_disaster.png"
    )

    task = (
        "Disaster Bearing Bodies Recognition"
    )

    question = (
        "What categories of objects have "
        "sustained damage in the affected area?"
    )

    results = retrieve_multimodal(
        image_path=image,
        question=question,
        task=task,
        text_top_k=5,
        image_top_k=5,
    )

    print()
    print("=" * 70)
    print("MULTIMODAL RETRIEVAL")
    print("=" * 70)

    print()
    print("TEXT EVIDENCE")
    print("-" * 70)

    for i, result in enumerate(
        results["text_results"],
        start=1,
    ):

        payload = result.payload

        print(
            f"\n[{i}] Score: "
            f"{result.score:.4f}"
        )

        print(
            "Question:",
            payload.get("question")
        )

        print(
            "Historical answer:",
            payload.get("answer")
        )

        print(
            "Source:",
            payload.get("source")
        )

    print()
    print()
    print("IMAGE EVIDENCE")
    print("-" * 70)

    for i, result in enumerate(
        results["image_results"],
        start=1,
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