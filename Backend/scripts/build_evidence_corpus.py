#!/usr/bin/env python3
"""
CLI Tool: Generate Standardized Multimodal Evidence Corpus for DisasterMind

Usage:
    python scripts/build_evidence_corpus.py
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure Backend/ is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.corpus_service import EvidenceCorpusGenerator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct standardized evidence corpus from DisasterM3 QA and retrieval sources."
    )
    parser.add_argument(
        "--qa-dataset",
        type=str,
        default="data/disasterm3/disasterm3_qa.json",
        help="Path to DisasterM3 QA dataset",
    )
    parser.add_argument(
        "--dev-corpus",
        type=str,
        default="data/disasterm3/retrieval/dev_corpus.json",
        help="Path to development retrieval corpus",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/disasterm3/retrieval/standardized_corpus.json",
        help="Destination path for standardized evidence corpus JSON",
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default="data/reports/corpus_statistics.json",
        help="Destination path for corpus statistics report JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    qa_path = Path(args.qa_dataset)
    if not qa_path.is_absolute():
        qa_path = (BACKEND_DIR / qa_path).resolve()

    dev_path = Path(args.dev_corpus)
    if not dev_path.is_absolute():
        dev_path = (BACKEND_DIR / dev_path).resolve()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (BACKEND_DIR / output_path).resolve()

    stats_path = Path(args.stats_output)
    if not stats_path.is_absolute():
        stats_path = (BACKEND_DIR / stats_path).resolve()

    print("=" * 70)
    print("DISASTERM3 STANDARDIZED EVIDENCE CORPUS GENERATOR")
    print("=" * 70)
    print(f"QA dataset       : {qa_path}")
    print(f"Dev corpus       : {dev_path}")
    print(f"Output path      : {output_path}")
    print(f"Statistics path  : {stats_path}")
    print("-" * 70)

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_records = json.load(f)

    dev_records = []
    if dev_path.exists():
        with open(dev_path, "r", encoding="utf-8") as f:
            dev_records = json.load(f)

    generator = EvidenceCorpusGenerator(base_dir=BACKEND_DIR)
    corpus, stats = generator.build_corpus(qa_records, dev_records)

    saved_corpus_path = generator.save_corpus(corpus, output_path)
    saved_stats_path = generator.save_statistics(stats, stats_path)

    print(f"Successfully generated {stats.total_records} evidence records.")
    print(f"  Corpus saved to      : {saved_corpus_path}")
    print(f"  Statistics saved to  : {saved_stats_path}")

    print("\nPROVENANCE DISTRIBUTION:")
    for src, count in sorted(stats.source_type_distribution.items(), key=lambda x: -x[1]):
        print(f"  - {src:<25}: {count}")

    print("\nDISASTER TYPE DISTRIBUTION:")
    for d_type, count in sorted(stats.disaster_type_distribution.items(), key=lambda x: -x[1]):
        print(f"  - {d_type:<25}: {count}")

    print("\nMODALITY DISTRIBUTION:")
    for mod, count in sorted(stats.modality_distribution.items(), key=lambda x: -x[1]):
        print(f"  - {mod:<25}: {count}")

    print(f"\nTotal visual assets referenced: {stats.total_images_referenced}")
    print("=" * 70)


if __name__ == "__main__":
    main()
