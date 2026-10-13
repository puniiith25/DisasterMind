import json
from pathlib import Path


QA_FILE = Path(
    "data/disasterm3/disasterm3_qa.json"
)

OUTPUT_FILE = Path(
    "data/disasterm3/retrieval/image_corpus.json"
)


def main():

    with open(QA_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    images = {}

    for item in records:

        paths = [
            item.get("pre_image"),
            item.get("post_image"),
            item.get("image")
        ]

        for path in paths:

            if not path:
                continue

            images[path] = {
                "image_path": path,
                "source": "DisasterM3",
                "source_type": "historical",
                "task": item.get("task")
            }

    corpus = list(images.values())

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

    print(f"Unique images: {len(corpus)}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()