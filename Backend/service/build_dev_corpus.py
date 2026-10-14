import json
from pathlib import Path


BENCHMARK_FILE = Path(
    "data/disasterm3/DisasterM3_Bench/benchmark_release.json"
)

EVAL_FILE = Path(
    "data/disasterm3/disasterm3_qa.json"
)

OUTPUT_FILE = Path(
    "data/disasterm3/retrieval/dev_corpus.json"
)


def normalize_question(value):

    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(str(x) for x in value)

    return str(value).strip()


def main():

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        evaluation = json.load(f)

    # Build fingerprints for evaluation records
    evaluation_keys = set()

    for item in evaluation:

        key = (
            normalize_question(item.get("question")),
            item.get("pre_image"),
            item.get("post_image"),
            item.get("image")
        )

        evaluation_keys.add(key)

    corpus = []

    for item in benchmark:

        question = normalize_question(
            item.get("prompts")
        )

        key = (
            question,
            item.get("pre_image_path"),
            item.get("post_image_path"),
            item.get("image_path")
        )

        # Do not use evaluation records
        if key in evaluation_keys:
            continue

        if not question:
            continue

        corpus.append({
            "id": f"retrieval_{len(corpus):05d}",
            "task": item.get("task"),
            "question": question,
            "answer": item.get("ground_truth"),
            "source": "DisasterM3",
            "source_type": "historical"
        })

        if len(corpus) >= 500:
            break

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            corpus,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"Created corpus: {len(corpus)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()