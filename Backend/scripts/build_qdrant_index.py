#!/usr/bin/env python3
"""
CLI Tool: Build Qdrant Dense Text Index from Evidence Corpus

Usage:
    python scripts/build_qdrant_index.py
    python scripts/build_qdrant_index.py --limit 100 --batch-size 16
    python scripts/build_qdrant_index.py --corpus data/disasterm3/retrieval/standardized_corpus.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure Backend/ is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.qdrant_service import QdrantIndexService


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Qdrant dense text index from DisasterM3 evidence corpus."
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/disasterm3/retrieval/standardized_corpus.json",
        help="Path to standardized_corpus.json",
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default="data/qdrant",
        help="Qdrant local storage directory",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="disasterm3_text",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="BGE-M3 embedding batch size",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max records to index (defaults to all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Embedding device: 'auto', 'mps', 'cpu'",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default="data/reports/qdrant_index_report.json",
        help="Path to save the JSON index build report",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.is_absolute():
        corpus_path = (BACKEND_DIR / corpus_path).resolve()

    qdrant_path = Path(args.qdrant_path)
    if not qdrant_path.is_absolute():
        qdrant_path = (BACKEND_DIR / qdrant_path).resolve()

    report_path = Path(args.report_out)
    if not report_path.is_absolute():
        report_path = (BACKEND_DIR / report_path).resolve()

    if not corpus_path.exists():
        print(f"[ERROR] Corpus not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    # Optional limit: slice corpus file
    if args.limit:
        with open(corpus_path, "r", encoding="utf-8") as f:
            records = json.load(f)[: args.limit]
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(records, tmp, ensure_ascii=False, indent=2)
        tmp.close()
        corpus_path_actual = Path(tmp.name)
    else:
        corpus_path_actual = corpus_path

    print("=" * 70)
    print("QDRANT DENSE TEXT INDEX BUILD")
    print("=" * 70)
    print(f"Corpus     : {corpus_path}")
    print(f"Collection : {args.collection}")
    print(f"Qdrant dir : {qdrant_path}")
    print(f"Device     : {args.device}")
    print(f"Batch size : {args.batch_size}")
    print("-" * 70)

    from app.services.embedding_service import BGEM3EmbeddingService

    embedding_service = BGEM3EmbeddingService(
        device=args.device,
        cache_dir=BACKEND_DIR / "data/cache/embeddings",
    )

    service = QdrantIndexService(
        qdrant_path=qdrant_path,
        collection_name=args.collection,
        embedding_service=embedding_service,
    )

    report = service.build_index(
        corpus_path=corpus_path_actual,
        batch_size=args.batch_size,
        recreate=True,
    )

    if args.limit:
        import os
        os.unlink(corpus_path_actual)

    print("\nINDEX BUILD RESULTS:")
    print(f"  Total records   : {report.total_records}")
    print(f"  Indexed vectors : {report.indexed_count}")
    print(f"  Skipped         : {report.skipped_count}")
    print(f"  Duration        : {report.latency_seconds:.2f} s")
    print(f"  Throughput      : {report.throughput_items_per_sec:.1f} items/s")
    print(f"  Cache hit rate  : {report.cache_hit_rate:.1%}")
    print(f"  Status          : {report.status}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2, exclude={"indexed_points"}))

    print(f"\nReport saved to: {report_path}")
    print("=" * 70)

    # Quick sanity retrieval
    print("\nSANITY RETRIEVAL TEST:")
    sample_query = "What damage was caused to buildings in the earthquake zone?"
    response = service.retrieve(sample_query, top_k=3)
    print(f"  Query   : {sample_query}")
    print(f"  Latency : {response.latency_ms:.1f} ms")
    for r in response.results:
        print(f"  [{r.rank}] score={r.score:.4f} | {r.task_type} | {r.evidence_id}")
    print("=" * 70)


if __name__ == "__main__":
    main()
