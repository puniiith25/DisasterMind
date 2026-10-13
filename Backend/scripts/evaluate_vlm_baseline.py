#!/usr/bin/env python3
"""
CLI Evaluation: Qwen2.5-VL Vision-Language Baseline Benchmark

Usage:
    python scripts/evaluate_vlm_baseline.py --samples 5 --output-report data/reports/vlm_baseline_evaluation.json
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

from app.schemas.vlm import (
    VLMBenchmarkReport,
    VLMBenchmarkSampleResult,
    VLMInferenceRequest,
)
from app.services.vlm_service import QwenVLMService


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-VL baseline model on DisasterM3 samples."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/disasterm3/disasterm3_qa.json",
        help="Path to DisasterM3 dataset JSON",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of representative samples to evaluate",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="data/reports/vlm_baseline_evaluation.json",
        help="Destination JSON for baseline benchmark results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on ('auto', 'mps', 'cpu', 'cuda')",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max generation tokens",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (BACKEND_DIR / dataset_path).resolve()

    report_path = Path(args.output_report)
    if not report_path.is_absolute():
        report_path = (BACKEND_DIR / report_path).resolve()

    if not dataset_path.exists():
        print(f"[ERROR] Dataset file not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Select representative samples covering different tasks
    selected_samples = []
    seen_tasks = set()
    for r in records:
        task = r.get("task", "")
        if task not in seen_tasks and len(selected_samples) < args.samples:
            selected_samples.append(r)
            seen_tasks.add(task)

    # Fill remaining if needed
    for r in records:
        if len(selected_samples) >= args.samples:
            break
        if r not in selected_samples:
            selected_samples.append(r)

    print("=" * 70)
    print("QWEN2.5-VL BASELINE EVALUATION")
    print("=" * 70)
    print(f"Dataset      : {dataset_path}")
    print(f"Sample count : {len(selected_samples)}")
    print(f"Device target: {args.device}")
    print("-" * 70)

    service = QwenVLMService(device=args.device)

    print("Loading model and processor...")
    mem_before = get_process_memory_mb()
    load_start = time.perf_counter()
    service.load_model()
    load_time = time.perf_counter() - load_start
    mem_after = get_process_memory_mb()

    print(f"Model loaded in {load_time:.2f}s (RAM: {mem_after - mem_before:.1f} MB added, total: {mem_after:.1f} MB)")
    print(f"Active device: {service.device}")
    print("-" * 70)

    sample_results = []
    total_latency = 0.0

    for i, s in enumerate(selected_samples, start=1):
        s_id = s.get("id", f"sample_{i}")
        task = s.get("task", "Unknown")
        question = s.get("question", "")
        ground_truth = str(s.get("answer", ""))

        pre_img = s.get("pre_image")
        post_img = s.get("post_image")
        img = s.get("image")

        # Resolve paths relative to BACKEND_DIR
        def resolve_p(p):
            if not p:
                return None
            path_obj = Path(p)
            return str((BACKEND_DIR / path_obj).resolve()) if not path_obj.is_absolute() else str(path_obj)

        req = VLMInferenceRequest(
            question=question,
            pre_image_path=resolve_p(pre_img),
            post_image_path=resolve_p(post_img),
            image_path=resolve_p(img),
            max_new_tokens=args.max_tokens,
        )

        resp = service.generate(req)
        mem_curr = get_process_memory_mb()
        total_latency += resp.latency_ms

        print(f"[{i}/{len(selected_samples)}] ID: {s_id} | Task: {task}")
        print(f"    Question    : {question}")
        print(f"    Ground Truth: {ground_truth}")
        print(f"    Prediction  : {resp.prediction}")
        print(f"    Latency     : {resp.latency_ms:.1f} ms | Status: {resp.status}")
        print("-" * 70)

        sample_results.append(
            VLMBenchmarkSampleResult(
                sample_id=s_id,
                task=task,
                question=question,
                ground_truth=ground_truth,
                prediction=resp.prediction,
                latency_ms=resp.latency_ms,
                memory_mb=mem_curr,
                pre_image=pre_img,
                post_image=post_img,
                image=img,
            )
        )

    avg_latency = total_latency / len(sample_results) if sample_results else 0.0
    peak_mem = get_process_memory_mb()

    report = VLMBenchmarkReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_id=service.model_id,
        device=service.device,
        load_time_seconds=load_time,
        samples_count=len(sample_results),
        avg_latency_ms=avg_latency,
        peak_memory_mb=peak_mem,
        samples=sample_results,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    print("=" * 70)
    print("BENCHMARK COMPLETED")
    print(f"  Avg Latency  : {avg_latency:.1f} ms")
    print(f"  Peak Memory  : {peak_mem:.1f} MB")
    print(f"  Report saved : {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
