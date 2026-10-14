#!/usr/bin/env python3
"""
CLI Tool: Run Multimodal RRF Retrieval (DM-06)

Usage:
    python scripts/run_multimodal_retrieval.py --question "Earthquake damage assessment"
    python scripts/run_multimodal_retrieval.py --question "Flood extent" --image data/sample.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure Backend/ is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.multimodal_service import (
    MultimodalRetrievalRequest,
    get_multimodal_service,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multimodal reciprocal rank fusion retrieval."
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What disaster damage occurred?",
        help="Query question text",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to query image",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        help="Optional disaster task category",
    )
    parser.add_argument(
        "--text-top-k",
        type=int,
        default=5,
        help="Top-K text results from Qdrant",
    )
    parser.add_argument(
        "--image-top-k",
        type=int,
        default=3,
        help="Top-K image results from CLIP",
    )
    parser.add_argument(
        "--fusion-top-k",
        type=int,
        default=5,
        help="Top-K fused results after RRF",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\n🔍 Running Multimodal Retrieval...")
    print(f"  Question:     {args.question}")
    print(f"  Image:        {args.image or 'None (text-only)'}")
    print(f"  Fusion Top-K: {args.fusion_top_k}\n")

    service = get_multimodal_service()
    request = MultimodalRetrievalRequest(
        question=args.question,
        image_path=args.image,
        task_type=args.task_type,
        text_top_k=args.text_top_k,
        image_top_k=args.image_top_k,
        fusion_top_k=args.fusion_top_k,
    )

    response = service.retrieve(request)

    print(f"⚡ Retrieval finished in {response.latency_ms:.1f} ms")
    print(f"  Fusion Mode:        {response.fusion_mode}")
    print(f"  Text Results:       {response.text_results_count}")
    print(f"  Image Results:      {response.image_results_count}")
    print(f"  Fused Count:        {len(response.fused_results)}\n")

    print("=" * 70)
    print("RANKED FUSED EVIDENCE RESULTS")
    print("=" * 70)
    for res in response.fused_results:
        print(f"\n[Rank {res.rank}] Evidence ID: {res.evidence_id} | Score: {res.rrf_score:.4f}")
        print(f"  Task:      {res.task_type} | Source: {res.source_type}")
        if res.disaster_type:
            print(f"  Disaster:  {res.disaster_type} | Location: {res.location or 'N/A'}")
        print(f"  Question:  {res.question}")
        print(f"  Answer:    {res.answer}")
    print("=" * 70)


if __name__ == "__main__":
    main()
