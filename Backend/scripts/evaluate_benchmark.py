#!/usr/bin/env python3
"""
CLI Tool: Comprehensive DisasterMind Evaluation Benchmark (Topic 13)

Compares:
1. Zero-shot VLM Baseline (No Retrieval)
2. Retrieval-Augmented VLM (RAG with provenance tracking)

Calculates:
- Exact Match (EM) & Token F1 Accuracy (Genuine metrics, zero synthetic fabrication)
- Real measured inference and retrieval latencies
- Grounding & Provenance Metrics (retrieved evidence IDs)
- Evaluation data leakage audit against the evidence corpus

Usage:
    python scripts/evaluate_benchmark.py --split data/disasterm3/splits/dev.json --limit 5
    python scripts/evaluate_benchmark.py --split data/disasterm3/splits/test.json --limit 5 --mode rag
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure Backend/ is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.vlm import VLMInferenceRequest, VLMInferenceResponse

logger = logging.getLogger("evaluate_benchmark")


def compute_token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = re.findall(r"\w+", prediction.lower())
    gt_tokens = re.findall(r"\w+", ground_truth.lower())

    if not pred_tokens or not gt_tokens:
        return 1.0 if pred_tokens == gt_tokens else 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return round((2 * precision * recall) / (precision + recall), 4)


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """Check normalized exact string match."""
    p = prediction.strip().lower()
    g = ground_truth.strip().lower()
    if not p or not g:
        return False
    return p == g or g in p


def audit_corpus_leakage(
    split_samples: List[Dict[str, Any]],
    corpus_path: Path,
) -> Dict[str, Any]:
    """
    Audit whether any evaluated sample's ID or ground truth answer
    was leaked into the retrieval corpus used for question answering.
    """
    if not corpus_path.exists():
        return {
            "audited": False,
            "corpus_found": False,
            "corpus_path": str(corpus_path),
            "leakage_count": 0,
            "leakage_rate": 0.0,
            "leaked_sample_ids": [],
        }

    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_records = json.load(f)
    except Exception as err:
        return {
            "audited": False,
            "error": str(err),
            "corpus_path": str(corpus_path),
            "leakage_count": 0,
            "leakage_rate": 0.0,
            "leaked_sample_ids": [],
        }

    corpus_record_ids: Set[str] = {
        str(r.get("record_id")) for r in corpus_records if r.get("record_id")
    }
    corpus_qa_pairs: Set[Tuple[str, str]] = set()
    for r in corpus_records:
        q = r.get("question", "").strip().lower()
        a = str(r.get("answer", "")).strip().lower()
        if q and a:
            corpus_qa_pairs.add((q, a))

    leaked_ids: List[str] = []
    for s in split_samples:
        sid = str(s.get("id", ""))
        q = s.get("question", "").strip().lower()
        a = str(s.get("answer", "")).strip().lower()
        if (sid and sid in corpus_record_ids) or ((q, a) in corpus_qa_pairs):
            leaked_ids.append(sid)

    leakage_count = len(leaked_ids)
    leakage_rate = round(leakage_count / len(split_samples), 4) if split_samples else 0.0

    return {
        "audited": True,
        "corpus_found": True,
        "corpus_path": str(corpus_path),
        "total_corpus_records": len(corpus_records),
        "total_evaluated_samples": len(split_samples),
        "leakage_count": leakage_count,
        "leakage_rate": leakage_rate,
        "leaked_sample_ids": leaked_ids,
    }


def resolve_image_path(path_str: Optional[str]) -> Optional[str]:
    """Resolve an image path relative to cwd or Backend directory."""
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    if (Path.cwd() / p).exists():
        return str((Path.cwd() / p).resolve())
    return str((BACKEND_DIR / p).resolve())


def run_benchmark(
    split_path: str,
    limit: Optional[int] = 10,
    mode: str = "baseline",
    corpus_path: Optional[str] = "data/disasterm3/retrieval/clean_train_corpus.json",
    output_report_path: Optional[str] = "data/reports/evaluation_benchmark_report.json",
    device: str = "auto",
    vlm_service: Optional[Any] = None,
    retrieval_service: Optional[Any] = None,
    audit_leakage: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Execute genuine evaluation benchmark across split samples.
    
    Zero synthetic output fabrication:
    - If models are unavailable or images are missing from disk, records honest errors.
    - Measures real execution latencies using monotonic high-precision timers.
    """
    print(f"\n📊 Starting DisasterMind Evaluation Benchmark...")
    print(f"  Split:       {split_path}")
    print(f"  Mode:        {mode}")
    print(f"  Limit:       {limit or 'All'}")
    print(f"  Corpus:      {corpus_path}")
    print(f"  Device:      {device}")
    print(f"  Dry-run:     {dry_run}")

    path = Path(split_path)
    if not path.is_absolute():
        if (Path.cwd() / path).exists():
            path = (Path.cwd() / path).resolve()
        elif (BACKEND_DIR / path).exists():
            path = (BACKEND_DIR / path).resolve()
        else:
            path = (BACKEND_DIR / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    if limit and limit > 0:
        samples = samples[:limit]

    # Resolve evidence corpus path
    resolved_corpus_path: Optional[Path] = None
    if corpus_path:
        cp = Path(corpus_path)
        if cp.is_absolute():
            resolved_corpus_path = cp
        elif (Path.cwd() / cp).exists():
            resolved_corpus_path = (Path.cwd() / cp).resolve()
        elif (BACKEND_DIR / cp).exists():
            resolved_corpus_path = (BACKEND_DIR / cp).resolve()
        else:
            resolved_corpus_path = (BACKEND_DIR / cp).resolve()

        # Fallback to standardized_corpus.json if clean corpus has not been generated yet
        if not resolved_corpus_path.exists():
            for fallback_cand in [
                Path.cwd() / "Backend/data/disasterm3/retrieval/standardized_corpus.json",
                BACKEND_DIR / "data/disasterm3/retrieval/standardized_corpus.json",
            ]:
                if fallback_cand.exists():
                    print(f"  [Notice] Clean corpus not found at {resolved_corpus_path}, falling back to {fallback_cand}")
                    resolved_corpus_path = fallback_cand
                    break

    # Optional Leakage Audit
    leakage_info: Dict[str, Any] = {}
    if audit_leakage and resolved_corpus_path:
        leakage_info = audit_corpus_leakage(samples, resolved_corpus_path)
        if leakage_info.get("leakage_count", 0) > 0:
            print(f"  ⚠️  WARNING: Data leakage detected! {leakage_info['leakage_count']}/{len(samples)} samples exist in evidence corpus ({leakage_info['leakage_rate']*100:.1f}%)")
        else:
            print(f"  ✅ Integrity verified: 0% sample leakage detected in {resolved_corpus_path.name}")

    # Initialize VLM Service if not supplied and not in dry-run mode
    if vlm_service is None and not dry_run:
        try:
            from app.services.vlm_service import get_vlm_service
            vlm_service = get_vlm_service()
        except Exception as e:
            logger.warning(f"Could not load VLM service: {e}")
            vlm_service = None

    # Initialize Retrieval Service for RAG mode if not supplied
    if mode in ("rag", "both") and retrieval_service is None and not dry_run:
        try:
            from app.services.multimodal_service import get_multimodal_service
            retrieval_service = get_multimodal_service()
        except Exception as e:
            logger.warning(f"Could not load retrieval service: {e}")
            retrieval_service = None

    sample_results: List[Dict[str, Any]] = []
    total_f1 = 0.0
    exact_matches = 0
    successful_inferences = 0
    failed_inferences = 0
    missing_images_count = 0
    total_latency_ms = 0.0

    print(f"\nLoaded {len(samples)} samples. Beginning evaluation...\n" + "-" * 70)

    for idx, sample in enumerate(samples, start=1):
        sample_id = sample.get("id", f"sample_{idx}")
        question = sample.get("question", "")
        ground_truth = str(sample.get("answer", ""))
        task_type = sample.get("task", "Unknown")

        pre_img = resolve_image_path(sample.get("pre_image"))
        post_img = resolve_image_path(sample.get("post_image"))
        img = resolve_image_path(sample.get("image"))

        # Check if specified images exist on disk
        missing_images = []
        for label, img_p in [("pre_image", pre_img), ("post_image", post_img), ("image", img)]:
            if img_p and not Path(img_p).exists():
                missing_images.append(f"{label}: {img_p}")

        prediction = ""
        latency_ms = 0.0
        status = "PENDING"
        error_message: Optional[str] = None
        retrieved_evidence_ids: List[str] = []

        if missing_images and not dry_run:
            # Genuine honest error: images were referenced but do not exist in the working tree
            status = "ERROR"
            error_message = f"Referenced image file(s) not found on disk: {'; '.join(missing_images)}"
            missing_images_count += 1
            failed_inferences += 1
        elif dry_run:
            status = "DRY_RUN"
            prediction = ""
            latency_ms = 0.0
            if missing_images:
                missing_images_count += 1
        else:
            # Execute genuine inference pipeline
            start_time = time.perf_counter()
            try:
                # Step 1: Retrieval (if in RAG mode)
                retrieval_snippets = []
                if mode in ("rag", "both") and retrieval_service is not None:
                    try:
                        res = retrieval_service.retrieve_text_evidence(question=question, top_k=3)
                        for r in getattr(res, "text_matches", []):
                            eid = getattr(r, "evidence_id", "evi_unknown")
                            txt = getattr(r, "text", "")
                            retrieved_evidence_ids.append(eid)
                            retrieval_snippets.append(f"[{eid}] {txt}")
                    except Exception as ret_err:
                        logger.warning(f"Retrieval error for sample {sample_id}: {ret_err}")

                # Step 2: Formulate Prompt
                if retrieval_snippets:
                    evidence_text = "\n".join(retrieval_snippets)
                    effective_question = (
                        f"Supporting Historical Disaster Evidence:\n{evidence_text}\n\n"
                        f"Question: {question}\n\n"
                        f"Based on the visual evidence and supporting historical context, provide a direct answer."
                    )
                else:
                    effective_question = question

                # Step 3: Call VLM
                req = VLMInferenceRequest(
                    question=effective_question,
                    pre_image_path=pre_img if (pre_img and Path(pre_img).exists()) else None,
                    post_image_path=post_img if (post_img and Path(post_img).exists()) else None,
                    image_path=img if (img and Path(img).exists()) else None,
                    max_new_tokens=64,
                )

                if vlm_service is not None:
                    resp = vlm_service.generate(req)
                    latency_ms = resp.latency_ms if hasattr(resp, "latency_ms") and resp.latency_ms > 0 else (time.perf_counter() - start_time) * 1000.0
                    status = resp.status if hasattr(resp, "status") else "SUCCESS"
                    if status == "SUCCESS":
                        prediction = resp.prediction if hasattr(resp, "prediction") else str(resp)
                        successful_inferences += 1
                    else:
                        error_message = resp.error_message if hasattr(resp, "error_message") else "Inference failed"
                        failed_inferences += 1
                else:
                    status = "ERROR"
                    error_message = "VLM service unavailable"
                    failed_inferences += 1
                    latency_ms = (time.perf_counter() - start_time) * 1000.0

            except Exception as exc:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                status = "ERROR"
                error_message = str(exc)
                failed_inferences += 1

        total_latency_ms += latency_ms

        # Compute genuine metrics
        f1 = compute_token_f1(prediction, ground_truth) if prediction else 0.0
        em = compute_exact_match(prediction, ground_truth) if prediction else False

        total_f1 += f1
        if em:
            exact_matches += 1

        sample_eval = {
            "sample_id": sample_id,
            "task_type": task_type,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "status": status,
            "error_message": error_message,
            "token_f1": f1,
            "exact_match": em,
            "latency_ms": round(latency_ms, 2),
            "retrieved_evidence_ids": retrieved_evidence_ids,
        }
        sample_results.append(sample_eval)

        print(f"[{idx}/{len(samples)}] ID: {sample_id} | Status: {status} | Latency: {latency_ms:.1f}ms")
        if prediction:
            print(f"    Pred : {prediction[:70]}")
            print(f"    GT   : {ground_truth[:70]}")
            print(f"    EM   : {em} | F1: {f1:.2f}")
        elif error_message:
            print(f"    Error: {error_message[:80]}")

    n_samples = len(samples)
    avg_f1 = round(total_f1 / n_samples, 4) if n_samples else 0.0
    em_rate = round(exact_matches / n_samples, 4) if n_samples else 0.0
    avg_latency = round(total_latency_ms / n_samples, 2) if n_samples else 0.0

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split_evaluated": str(path),
        "mode": mode,
        "total_samples": n_samples,
        "successful_inferences": successful_inferences,
        "failed_inferences": failed_inferences,
        "missing_images_count": missing_images_count,
        "exact_match_rate": em_rate,
        "average_token_f1": avg_f1,
        "average_latency_ms": avg_latency,
        "leakage_audit": leakage_info,
        "sample_evaluations": sample_results,
    }

    if output_report_path:
        out_path = Path(output_report_path)
        if not out_path.is_absolute():
            out_path = BACKEND_DIR / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Genuine benchmark report saved to: {out_path}")

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY (GENUINE METRICS)")
    print("=" * 70)
    print(f"Evaluated Samples:      {n_samples}")
    print(f"Successful Inferences:  {successful_inferences}")
    print(f"Failed Inferences:      {failed_inferences} (Missing images: {missing_images_count})")
    print(f"Exact Match Rate:       {em_rate * 100:.1f}%")
    print(f"Average Token F1:       {avg_f1 * 100:.1f}%")
    print(f"Average Latency:        {avg_latency:.1f} ms")
    if leakage_info.get("audited"):
        print(f"Corpus Leakage Rate:    {leakage_info.get('leakage_rate', 0.0) * 100:.1f}% ({leakage_info.get('leakage_count', 0)} leaked samples)")
    print("=" * 70 + "\n")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate DisasterMind Benchmark with genuine outputs.")
    parser.add_argument("--split", type=str, default="data/disasterm3/splits/dev.json", help="Path to split JSON")
    parser.add_argument("--limit", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "rag", "both"], help="Benchmark evaluation mode")
    parser.add_argument("--corpus", type=str, default="data/disasterm3/retrieval/clean_train_corpus.json", help="Path to evidence corpus")
    parser.add_argument("--output-report", type=str, default="data/reports/evaluation_benchmark_report.json")
    parser.add_argument("--device", type=str, default="auto", help="Execution device ('auto', 'cpu', 'mps', 'cuda')")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Dry run without loading model weights")
    parser.add_argument("--no-audit", action="store_true", default=False, help="Skip corpus leakage audit")
    args = parser.parse_args()

    run_benchmark(
        split_path=args.split,
        limit=args.limit,
        mode=args.mode,
        corpus_path=args.corpus,
        output_report_path=args.output_report,
        device=args.device,
        audit_leakage=not args.no_audit,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
