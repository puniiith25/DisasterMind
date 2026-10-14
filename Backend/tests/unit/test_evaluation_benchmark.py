"""
Unit tests for Topic 13: Evaluation Benchmark Metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from scripts.evaluate_benchmark import (
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

    out_report = tmp_path / "report.json"
    report = run_benchmark(
        split_path=str(mock_split),
        limit=2,
        output_report_path=str(out_report),
    )

    assert report["total_samples"] == 2
    assert report["exact_match_rate"] > 0.0
    assert report["average_token_f1"] > 0.0
    assert out_report.exists()
