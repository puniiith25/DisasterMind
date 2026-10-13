"""
Unit tests for QdrantIndexService (DM-05).

All Qdrant interactions are fully mocked — no network or disk I/O needed.
BGE-M3 model calls are mocked through BGEM3EmbeddingService.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.schemas.retrieval import (
    CollectionStatus,
    IndexBuildReport,
    RetrievalResponse,
    RetrievalResult,
)
from app.services.qdrant_service import QdrantIndexService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DUMMY_RECORDS: List[Dict[str, Any]] = [
    {
        "evidence_id": "evi_001",
        "record_id": "dm3_00001",
        "text": "Task: Damage Assessment\nQuestion: How severe is the damage?\nHistorical answer: Severe structural damage",
        "task_type": "Damage Assessment",
        "question": "How severe is the damage?",
        "answer": "Severe structural damage",
        "disaster_type": "Earthquake",
        "location": "Turkey",
        "source_type": "qa",
        "images": [],
    },
    {
        "evidence_id": "evi_002",
        "record_id": "dm3_00002",
        "text": "Task: Flood Detection\nQuestion: Is flooding present?\nHistorical answer: Yes, widespread flooding",
        "task_type": "Flood Detection",
        "question": "Is flooding present?",
        "answer": "Yes, widespread flooding",
        "disaster_type": "Flood",
        "location": "Libya",
        "source_type": "retrieval",
        "images": [],
    },
    {
        "evidence_id": "evi_003",
        "record_id": "dm3_00003",
        "text": "Task: Fire Detection\nQuestion: Is there fire damage?\nHistorical answer: Active fire spotted",
        "task_type": "Fire Detection",
        "question": "Is there fire damage?",
        "answer": "Active fire spotted",
        "disaster_type": "Wildfire",
        "location": "California",
        "source_type": "qa",
        "images": [],
    },
]

DUMMY_VECTOR = [0.1] * 1024


def make_mock_embedding_service(dimension: int = 1024) -> MagicMock:
    """Return a mocked BGEM3EmbeddingService that returns unit vectors."""
    from app.schemas.embedding import EmbeddingBatchResult, EmbeddingMetadata, EmbeddingRepresentation

    mock_svc = MagicMock()
    mock_svc.model_id = "BAAI/bge-m3"
    mock_svc.dimension = dimension
    mock_svc.device = "cpu"

    def fake_embed_batch(texts, batch_size=16):
        vectors = [[0.1] * dimension for _ in texts]
        meta = EmbeddingMetadata(
            model_id="BAAI/bge-m3",
            dimension=dimension,
            representation=EmbeddingRepresentation.DENSE,
            normalized=True,
            device="cpu",
        )
        result = EmbeddingBatchResult(
            count=len(texts),
            cached_count=len(texts),
            computed_count=0,
            latency_ms=10.0,
            metadata=meta,
        )
        return vectors, result

    mock_svc.embed_batch.side_effect = fake_embed_batch
    mock_svc.embed_text.return_value = [0.1] * dimension
    return mock_svc


def make_qdrant_service(tmp_path: Path, dimension: int = 1024) -> QdrantIndexService:
    """Build a QdrantIndexService pointing to a temp directory with mock embeddings."""
    mock_emb = make_mock_embedding_service(dimension)
    svc = QdrantIndexService(
        qdrant_path=tmp_path / "qdrant",
        collection_name="test_collection",
        vector_size=dimension,
        embedding_service=mock_emb,
    )
    return svc


def write_corpus(tmp_path: Path) -> Path:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(DUMMY_RECORDS), encoding="utf-8")
    return corpus_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQdrantIndexService:

    def test_build_index_returns_report(self, tmp_path):
        """build_index() returns a valid IndexBuildReport with correct counts."""
        svc = make_qdrant_service(tmp_path)
        corpus_path = write_corpus(tmp_path)

        report = svc.build_index(corpus_path=corpus_path, batch_size=2)

        assert isinstance(report, IndexBuildReport)
        assert report.indexed_count == len(DUMMY_RECORDS)
        assert report.skipped_count == 0
        assert report.total_records == len(DUMMY_RECORDS)
        assert report.collection_name == "test_collection"

    def test_build_index_collection_created(self, tmp_path):
        """After build_index(), the Qdrant collection is created on disk."""
        svc = make_qdrant_service(tmp_path)
        corpus_path = write_corpus(tmp_path)

        report = svc.build_index(corpus_path=corpus_path)

        assert report.status == CollectionStatus.CREATED
        info = svc.collection_info()
        assert info["exists"] is True
        assert info["vectors_count"] == len(DUMMY_RECORDS)

    def test_build_index_idempotent_rebuild(self, tmp_path):
        """Calling build_index() twice with recreate=True replaces the collection cleanly."""
        svc = make_qdrant_service(tmp_path)
        corpus_path = write_corpus(tmp_path)

        report1 = svc.build_index(corpus_path=corpus_path)
        report2 = svc.build_index(corpus_path=corpus_path, recreate=True)

        assert report2.indexed_count == len(DUMMY_RECORDS)
        info = svc.collection_info()
        assert info["vectors_count"] == len(DUMMY_RECORDS)

    def test_build_index_missing_corpus_raises(self, tmp_path):
        """build_index() raises FileNotFoundError for a missing corpus path."""
        svc = make_qdrant_service(tmp_path)
        with pytest.raises(FileNotFoundError):
            svc.build_index(corpus_path=tmp_path / "nonexistent.json")

    def test_retrieve_returns_ranked_results(self, tmp_path):
        """retrieve() returns a RetrievalResponse with top_k ranked results."""
        svc = make_qdrant_service(tmp_path)
        corpus_path = write_corpus(tmp_path)
        svc.build_index(corpus_path=corpus_path)

        response = svc.retrieve(query="earthquake damage buildings", top_k=2)

        assert isinstance(response, RetrievalResponse)
        assert response.query == "earthquake damage buildings"
        assert response.top_k == 2
        assert len(response.results) <= 2
        assert response.latency_ms >= 0.0

        for r in response.results:
            assert isinstance(r, RetrievalResult)
            # Cosine similarity can exceed 1.0 slightly due to float32 precision
            assert -0.01 <= r.score <= 1.01, f"Unexpected score: {r.score}"

    def test_retrieve_results_ranked_descending(self, tmp_path):
        """Retrieval results are ordered by descending similarity score."""
        svc = make_qdrant_service(tmp_path)
        corpus_path = write_corpus(tmp_path)
        svc.build_index(corpus_path=corpus_path)

        response = svc.retrieve("flooding in Libya", top_k=3)
        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True), "Results must be in descending score order"

    def test_collection_info_before_build(self, tmp_path):
        """collection_info() returns exists=False before any index is built."""
        svc = make_qdrant_service(tmp_path)
        info = svc.collection_info()
        assert info["exists"] is False

    def test_collection_info_after_build(self, tmp_path):
        """collection_info() returns correct metadata after a successful build."""
        svc = make_qdrant_service(tmp_path)
        corpus_path = write_corpus(tmp_path)
        svc.build_index(corpus_path=corpus_path)

        info = svc.collection_info()
        assert info["exists"] is True
        assert info["vectors_count"] == len(DUMMY_RECORDS)

    def test_indexed_points_provenance(self, tmp_path):
        """Every indexed point carries correct evidence_id and record_id."""
        svc = make_qdrant_service(tmp_path)
        corpus_path = write_corpus(tmp_path)

        report = svc.build_index(corpus_path=corpus_path)

        evidence_ids = {p.evidence_id for p in report.indexed_points}
        record_ids = {p.record_id for p in report.indexed_points}

        for rec in DUMMY_RECORDS:
            assert rec["evidence_id"] in evidence_ids
            assert rec["record_id"] in record_ids
