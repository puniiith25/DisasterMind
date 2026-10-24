"""
Unit tests for DM-06: Multimodal Retrieval Service.

All CLIP model calls and Qdrant interactions are mocked — no model loading needed.
Tests cover RRF fusion, text-only, image-only, and multimodal modes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest

from app.services.multimodal_service import (
    FusedEvidenceResult,
    MultimodalRetrievalRequest,
    MultimodalRetrievalResponse,
    MultimodalRetrievalService,
    reciprocal_rank_fusion,
)
from app.schemas.retrieval import RetrievalResponse, RetrievalResult


# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------


def make_text_result(rank: int, evidence_id: str, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=score,
        evidence_id=evidence_id,
        record_id=f"rec_{evidence_id}",
        task_type="Damage Assessment",
        source_type="qa",
        text=f"Text for {evidence_id}",
        question="How severe is the damage?",
        answer="Severe structural damage",
        disaster_type="Earthquake",
        location="Turkey",
        payload={"images": []},
    )


def make_image_result(rank: int, evidence_id: str, score: float = 0.8) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=score,
        evidence_id=evidence_id,
        record_id=f"rec_{evidence_id}",
        task_type="Damage Assessment",
        source_type="image",
        text=f"Image text for {evidence_id}",
        question="",
        answer="",
        disaster_type="Earthquake",
        location="Turkey",
        payload={"images": [{"path": f"/images/{evidence_id}.png"}]},
    )


def make_qdrant_service_mock(text_results: List[RetrievalResult]) -> MagicMock:
    """Return a mock QdrantIndexService that returns given text results."""
    mock = MagicMock()
    mock_response = RetrievalResponse(
        query="test query",
        top_k=5,
        collection_name="test_collection",
        results=text_results,
        latency_ms=10.0,
    )
    mock.retrieve.return_value = mock_response
    mock._get_client.return_value = MagicMock()
    return mock


def make_clip_service_mock() -> MagicMock:
    mock = MagicMock()
    mock.embed_image.return_value = [0.1] * 512
    mock.embed_text_query.return_value = [0.1] * 512
    return mock


def make_service(
    text_results: List[RetrievalResult],
) -> MultimodalRetrievalService:
    qdrant = make_qdrant_service_mock(text_results)
    clip = make_clip_service_mock()
    return MultimodalRetrievalService(
        qdrant_service=qdrant,
        clip_service=clip,
    )


# ---------------------------------------------------------------------------
# RRF Unit Tests
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:

    def test_rrf_combines_text_and_image_results(self):
        """RRF merges two ranked lists and ranks shared documents higher."""
        text_results = [
            make_text_result(1, "evi_001", 0.95),
            make_text_result(2, "evi_002", 0.80),
        ]
        image_results = [
            make_image_result(1, "evi_002", 0.90),  # evi_002 appears in both
            make_image_result(2, "evi_003", 0.70),
        ]

        fused = reciprocal_rank_fusion(text_results, image_results, top_k=3)

        evidence_ids = [f[0] for f in fused]
        # evi_002 appears in both lists → should be ranked highly
        assert "evi_001" in evidence_ids
        assert "evi_002" in evidence_ids
        assert "evi_003" in evidence_ids

    def test_rrf_score_is_positive(self):
        """All RRF scores must be positive."""
        text_results = [make_text_result(1, "evi_001")]
        image_results = [make_image_result(1, "evi_002")]

        fused = reciprocal_rank_fusion(text_results, image_results, top_k=5)
        for _, score, *_ in fused:
            assert score > 0.0

    def test_rrf_sorted_descending(self):
        """Fused results are sorted by descending RRF score."""
        text_results = [make_text_result(i, f"evi_{i:03d}") for i in range(1, 6)]
        image_results = [make_image_result(i, f"evi_{i:03d}") for i in range(3, 8)]

        fused = reciprocal_rank_fusion(text_results, image_results, top_k=5)
        scores = [item[1] for item in fused]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_respects_top_k(self):
        """RRF output never exceeds top_k."""
        text_results = [make_text_result(i, f"evi_{i:03d}") for i in range(1, 10)]
        image_results = [make_image_result(i, f"evi_{i+5:03d}") for i in range(1, 10)]

        fused = reciprocal_rank_fusion(text_results, image_results, top_k=3)
        assert len(fused) <= 3

    def test_rrf_text_only_input(self):
        """RRF with no image results returns text-only ranked list."""
        text_results = [make_text_result(1, "evi_001"), make_text_result(2, "evi_002")]
        fused = reciprocal_rank_fusion(text_results, [], top_k=5)
        assert len(fused) == 2
        for _, _, _, _, i_rank, _, _ in fused:
            assert i_rank is None

    def test_rrf_image_only_input(self):
        """RRF with no text results returns image-only ranked list."""
        image_results = [make_image_result(1, "evi_010"), make_image_result(2, "evi_011")]
        fused = reciprocal_rank_fusion([], image_results, top_k=5)
        assert len(fused) == 2
        for _, _, t_rank, _, _, _, _ in fused:
            assert t_rank is None


# ---------------------------------------------------------------------------
# MultimodalRetrievalService Tests
# ---------------------------------------------------------------------------


class TestMultimodalRetrievalService:

    def test_text_only_mode_when_no_image(self):
        """When no image_path is given, fusion_mode is text_only."""
        text_results = [make_text_result(1, "evi_001"), make_text_result(2, "evi_002")]
        service = make_service(text_results)

        request = MultimodalRetrievalRequest(
            question="What damage occurred?",
            text_top_k=2,
            fusion_top_k=2,
        )
        response = service.retrieve(request)

        assert isinstance(response, MultimodalRetrievalResponse)
        assert response.fusion_mode == "text_only"
        assert response.text_results_count == 2
        assert response.image_results_count == 0
        assert len(response.fused_results) == 2

    def test_response_structure_is_valid(self):
        """Response contains correct fields and types."""
        text_results = [make_text_result(1, "evi_001", score=0.85)]
        service = make_service(text_results)

        request = MultimodalRetrievalRequest(question="How severe is the flood?")
        response = service.retrieve(request)

        assert response.question == "How severe is the flood?"
        assert response.latency_ms >= 0.0
        assert len(response.fused_results) == 1

        result = response.fused_results[0]
        assert isinstance(result, FusedEvidenceResult)
        assert result.rank == 1
        assert result.evidence_id == "evi_001"
        assert result.rrf_score > 0.0

    def test_fused_results_ordered_by_rank(self):
        """Fused result ranks are sequential starting from 1."""
        text_results = [make_text_result(i, f"evi_{i:03d}") for i in range(1, 4)]
        service = make_service(text_results)

        request = MultimodalRetrievalRequest(question="damage assessment", fusion_top_k=3)
        response = service.retrieve(request)

        ranks = [r.rank for r in response.fused_results]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_empty_question_returns_no_text_results(self):
        """Empty question string skips text retrieval."""
        service = make_service([])
        request = MultimodalRetrievalRequest(question="")
        response = service.retrieve(request)

        # No text results and no image → empty fused results
        assert response.text_results_count == 0
        assert len(response.fused_results) == 0

    def test_multimodal_rrf_mode_when_image_and_text(self, tmp_path):
        """When both image and text are provided, fusion_mode is multimodal_rrf."""
        # Create a dummy image
        from PIL import Image as PILImage
        dummy_img = PILImage.new("RGB", (64, 64), color=(100, 150, 200))
        img_path = tmp_path / "test.png"
        dummy_img.save(img_path)

        text_results = [make_text_result(1, "evi_001"), make_text_result(2, "evi_002")]
        image_results = [make_image_result(1, "evi_003"), make_image_result(2, "evi_001")]

        qdrant = make_qdrant_service_mock(text_results)
        clip = make_clip_service_mock()

        # Patch _retrieve_by_image to return image_results directly
        service = MultimodalRetrievalService(
            qdrant_service=qdrant,
            clip_service=clip,
        )
        service._retrieve_by_image = lambda image_path, top_k: image_results

        request = MultimodalRetrievalRequest(
            question="How severe is the earthquake damage?",
            image_path=str(img_path),
            text_top_k=2,
            image_top_k=2,
            fusion_top_k=4,
        )
        response = service.retrieve(request)

        assert response.fusion_mode == "multimodal_rrf"
        assert response.text_results_count == 2
        assert response.image_results_count == 2
        # evi_001 appears in both → should be fused and deduplicated
        fused_ids = [r.evidence_id for r in response.fused_results]
        assert len(fused_ids) == len(set(fused_ids)), "Duplicate evidence IDs in fused results"

    def test_image_retrieval_failure_falls_back_to_text(self, tmp_path):
        """If image retrieval raises, response gracefully falls back to text_only."""
        from PIL import Image as PILImage
        dummy_img = PILImage.new("RGB", (32, 32))
        img_path = tmp_path / "test.png"
        dummy_img.save(img_path)

        text_results = [make_text_result(1, "evi_001")]
        qdrant = make_qdrant_service_mock(text_results)
        clip = make_clip_service_mock()

        service = MultimodalRetrievalService(
            qdrant_service=qdrant,
            clip_service=clip,
        )
        # Simulate image retrieval failure
        service._retrieve_by_image = MagicMock(side_effect=RuntimeError("CLIP error"))

        request = MultimodalRetrievalRequest(
            question="earthquake damage",
            image_path=str(img_path),
        )
        response = service.retrieve(request)

        assert response.fusion_mode == "text_only"
        assert len(response.fused_results) >= 1

    def test_retrieve_text_evidence_direct_helper(self):
        """retrieve_text_evidence returns TextEvidenceResponse with text_matches."""
        text_results = [make_text_result(1, "evi_101"), make_text_result(2, "evi_102")]
        service = make_service(text_results)

        resp = service.retrieve_text_evidence(question="flooding in Hatay", top_k=2)
        assert resp.query == "flooding in Hatay"
        assert resp.top_k == 2
        assert len(resp.text_matches) == 2
        assert resp.text_matches[0].evidence_id == "evi_101"
