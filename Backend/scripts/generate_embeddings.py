#!/usr/bin/env python3
"""
CLI Tool: Generate & Benchmark BGE-M3 Dense Text Embeddings

Usage:
    python scripts/generate_embeddings.py --limit 100 --batch-size 16
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import psutil

# Ensure Backend/ is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.embedding import EmbeddingBenchmarkReport
from app.services.embedding_service import BGEM3EmbeddingService


def get_process_memory_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and benchmark dense BGE-M3 text embeddings with persistent caching."
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/disasterm3/retrieval/standardized_corpus.json",
        help="Path to evidence corpus JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum records to process (defaults to all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding generation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device ('auto', 'mps', 'cpu')",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default="data/reports/embedding_benchmark.json",
        help="Destination path for benchmark metrics report JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.is_absolute():
        corpus_path = (BACKEND_DIR / corpus_path).resolve()

    report_path = Path(args.report_out)
    if not report_path.is_absolute():
        report_path = (BACKEND_DIR / report_path).resolve()

    if not corpus_path.exists():
        print(f"[ERROR] Corpus file not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    with open(corpus_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if args.limit:
        records = records[: args.limit]

    texts = [r.get("text", "") for r in records]

    print("=" * 70)
    print("BGE-M3 EMBEDDING GENERATION & BENCHMARK")
    print("=" * 70)
    print(f"Corpus items : {len(texts)}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Device       : {args.device}")
    print("-" * 70)

    service = BGEM3EmbeddingService(device=args.device, cache_dir=BACKEND_DIR / "data/cache/embeddings")

    start_mem = get_process_memory_mb()
    start_time = time.perf_counter()

    vectors, batch_result = service.embed_batch(texts, batch_size=args.batch_size)

    total_time = time.perf_counter() - start_time
    end_mem = get_process_memory_mb()

    throughput = len(texts) / total_time if total_time > 0 else 0.0
    cache_hit_rate = batch_result.cached_count / len(texts) if texts else 0.0

    print("\nEMBEDDING RESULTS:")
    print(f"  Total Vectors    : {len(vectors)}")
    print(f"  Dimension        : {batch_result.metadata.dimension}")
    print(f"  Cache Hits       : {batch_result.cached_count}")
    print(f"  Computed Newly   : {batch_result.computed_count}")
    print(f"  Total Duration   : {total_time:.2f} s")
    print(f"  Throughput       : {throughput:.1f} items/sec")
    print(f"  Device Used      : {batch_result.metadata.device}")
    print(f"  Memory Footprint : {end_mem:.1f} MB (Delta: {end_mem - start_mem:+.1f} MB)")

    report = EmbeddingBenchmarkReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_id=service.model_id,
        device=service.device,
        dimension=service.dimension,
        representation=batch_result.metadata.representation.value,
        total_texts=len(texts),
        batch_size=args.batch_size,
        total_latency_seconds=total_time,
        throughput_items_per_sec=throughput,
        cache_hit_rate=cache_hit_rate,
        memory_mb=end_mem,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    print(f"\nReport saved to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
