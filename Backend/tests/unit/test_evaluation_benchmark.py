"""
Unit tests for Topic 13: Evaluation Benchmark Metrics and Research Integrity.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app.schemas.vlm import VLMInferenceResponse
from scripts.evaluate_benchmark import (
    audit_corpus_leakage,
    compute_exact_match,
    compute_token_f1,
    run_benchmark,
)


def test_compute_token_f1():
    assert compute_token_f1("45 collapsed buildings", "45 collapsed buildings") == 1.0
    assert compute_token_f1("45 collapsed buildings", "no damage") == 0.0
    f1 = compute_token_f1("There are 45 collapsed buildings detected.", "45 collapsed buildings")
    assert 0.5 <= f1 <= 1.0


def test_compute_exact_match():
    assert compute_exact_match("Earthquake", "earthquake") is True
    assert compute_exact_match("Severe earthquake in Turkey", "Earthquake") is True
    assert compute_exact_match("Tornado", "Hurricane") is False
    assert compute_exact_match("", "Earthquake") is False
    assert compute_exact_match("Earthquake", "") is False


def test_audit_corpus_leakage(tmp_path):
    mock_corpus = tmp_path / "mock_corpus.json"
    corpus_data = [
        {
            "evidence_id": "evi_001",
            "record_id": "dm3_001",
            "question": "What disaster occurred?",
            "answer": "Earthquake",
            "text": "Task: Type\nQuestion: What disaster occurred?\nObserved Historical Answer: Earthquake",
        },
        {
            "evidence_id": "evi_002",
            "record_id": "dm3_002",
            "question": "How many damaged?",
            "answer": "10",
            "text": "Task: Count\nQuestion: How many damaged?\nObserved Historical Answer: 10",
        },
    ]
    with open(mock_corpus, "w", encoding="utf-8") as f:
        json.dump(corpus_data, f)

    split_samples = [
        {"id": "dm3_001", "question": "What disaster occurred?", "answer": "Earthquake"},  # Leaked ID & pair
        {"id": "dm3_999", "question": "Is there flood?", "answer": "No"},                  # Clean sample
    ]

    leakage_info = audit_corpus_leakage(split_samples, mock_corpus)
    assert leakage_info["audited"] is True
    assert leakage_info["leakage_count"] == 1
    assert "dm3_001" in leakage_info["leaked_sample_ids"]
    assert leakage_info["leakage_rate"] == 0.5


def test_run_benchmark_mock_split(tmp_path):
    mock_split = tmp_path / "mock_dev.json"
    data = [
        {
            "id": "dm3_001",
            "task": "Disaster Type Recognition",
            "question": "What disaster occurred?",
            "answer": "Earthquake",
        },
        {
            "id": "dm3_002",
            "task": "Building Damage Counting",
            "question": "How many buildings collapsed?",
            "answer": "12",
        },
    ]
    with open(mock_split, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Inject mock VLM service to test execution pipeline without loading heavy weights
    mock_vlm = MagicMock()
    mock_vlm.generate.side_effect = [
        VLMInferenceResponse(
            prediction="Earthquake",
            latency_ms=12.5,
            status="SUCCESS",
            device="cpu",
            model_id="mock_qwen",
        ),
        VLMInferenceResponse(
            prediction="There are 12 collapsed buildings.",
            latency_ms=15.0,
            status="SUCCESS",
            device="cpu",
            model_id="mock_qwen",
        ),
    ]

    out_report = tmp_path / "report.json"
    report = run_benchmark(
        split_path=str(mock_split),
        limit=2,
        output_report_path=str(out_report),
        vlm_service=mock_vlm,
        corpus_path=None,
        audit_leakage=False,
    )

    assert report["total_samples"] == 2
    assert report["successful_inferences"] == 2
    assert report["failed_inferences"] == 0
    assert report["exact_match_rate"] == 1.0
    assert report["average_token_f1"] > 0.5
    assert report["average_latency_ms"] > 0.0
    assert out_report.exists()


def test_run_benchmark_honest_missing_images_error(tmp_path):
    """Ensure missing dataset images produce honest errors rather than fabricated answers."""
    mock_split = tmp_path / "mock_missing_img.json"
    data = [
        {
            "id": "dm3_missing",
            "task": "Building Damage Counting",
            "question": "How many buildings survived?",
            "answer": "5",
            "pre_image": "non_existent_folder/missing_pre.png",
            "post_image": "non_existent_folder/missing_post.png",
        }
    ]
    with open(mock_split, "w", encoding="utf-8") as f:
        json.dump(data, f)

    out_report = tmp_path / "report_err.json"
    report = run_benchmark(
        split_path=str(mock_split),
        limit=1,
        output_report_path=str(out_report),
        audit_leakage=False,
    )

    assert report["total_samples"] == 1
    assert report["missing_images_count"] == 1
    assert report["failed_inferences"] == 1
    assert report["successful_inferences"] == 0
    assert report["exact_match_rate"] == 0.0
    assert report["average_token_f1"] == 0.0

    eval_sample = report["sample_evaluations"][0]
    assert eval_sample["status"] == "ERROR"
    assert "not found on disk" in eval_sample["error_message"].lower()
    assert eval_sample["prediction"] == ""


def test_run_benchmark_dry_run(tmp_path):
    mock_split = tmp_path / "mock_dev.json"
    data = [
        {
            "id": "dm3_001",
            "task": "Disaster Type Recognition",
            "question": "What disaster occurred?",
            "answer": "Earthquake",
        }
    ]
    with open(mock_split, "w", encoding="utf-8") as f:
        json.dump(data, f)

    report = run_benchmark(
        split_path=str(mock_split),
        limit=1,
        dry_run=True,
        audit_leakage=False,
        output_report_path=str(tmp_path / "dry_run_report.json"),
    )

    assert report["total_samples"] == 1
    assert report["sample_evaluations"][0]["status"] == "DRY_RUN"
