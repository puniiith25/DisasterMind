#!/usr/bin/env python3
"""
CLI Tool: Run DisasterMind Multimodal RAG Pipeline (DM-07)

Usage:
    python scripts/run_rag_pipeline.py --question "How many buildings collapsed in the earthquake?"
    python scripts/run_rag_pipeline.py --question "Evaluate flood severity" --post-image data/post.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure Backend/ is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.rag import RAGRequest
from app.services.rag_service import get_rag_service


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run end-to-end multimodal DisasterMind RAG pipeline."
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What disaster damage occurred?",
        help="Disaster assessment query question",
    )
    parser.add_argument(
        "--pre-image",
        type=str,
        default=None,
        help="Optional path to pre-disaster image",
    )
    parser.add_argument(
        "--post-image",
        type=str,
        default=None,
        help="Optional path to post-disaster image",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to single observation image",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        help="Disaster evaluation task category",
    )
    parser.add_argument(
        "--text-top-k",
        type=int,
        default=5,
        help="Top-K text results to retrieve",
    )
    parser.add_argument(
        "--fusion-top-k",
        type=int,
        default=3,
        help="Top-K evidence snippets to inject into VLM context",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max tokens for VLM generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\n🚀 Launching DisasterMind RAG Pipeline...")
    print(f"  Question:     {args.question}")
    print(f"  Pre-Image:    {args.pre_image or 'None'}")
    print(f"  Post-Image:   {args.post_image or 'None'}")
    print(f"  Single Image: {args.image or 'None'}")
    print(f"  Top-K:        {args.fusion_top_k}\n")

    rag_service = get_rag_service()
    request = RAGRequest(
        question=args.question,
        task_type=args.task_type,
        pre_image_path=args.pre_image,
        post_image_path=args.post_image,
        image_path=args.image,
        text_top_k=args.text_top_k,
        fusion_top_k=args.fusion_top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        include_provenance=True,
    )

    response = rag_service.run(request)

    print("=" * 70)
    print("RAG PIPELINE RESULT")
    print("=" * 70)
    print(f"Status:             {response.status}")
    print(f"Retrieval Mode:     {response.retrieval_mode}")
    print(f"Evidence Injected:  {response.evidence_count} snippets ({response.context_chars} chars)")
    print(f"Retrieval Latency:  {response.retrieval_latency_ms:.1f} ms")
    print(f"VLM Latency:        {response.vlm_latency_ms:.1f} ms")
    print(f"Total Latency:      {response.total_latency_ms:.1f} ms")
    print(f"Device:             {response.device}")
    print(f"VLM Model:          {response.vlm_model_id}\n")

    if response.evidence_snippets:
        print("-" * 70)
        print("RETRIEVED EVIDENCE SNIPPETS:")
        for snip in response.evidence_snippets:
            print(f"  [{snip.rank}] ({snip.source_type}) {snip.evidence_id}: {snip.answer} (RRF: {snip.rrf_score:.4f})")
        print("-" * 70)

    print("\nPREDICTION:")
    print(response.prediction)
    print("=" * 70)


if __name__ == "__main__":
    main()
