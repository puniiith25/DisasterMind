"""
Unit tests for DM-07: RAG Pipeline Service.

Tests cover:
- Context builder: evidence assembly, char budget enforcement
- RAGPipelineService: full pipeline with mocked retrieval + VLM
- Edge cases: no evidence, error recovery, provenance inclusion
"""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock

import pytest

from app.schemas.rag import (
    EvidenceSnippet,
    RAGContext,
    RAGRequest,
    RAGResponse,
)
from app.services.multimodal_service import (
    FusedEvidenceResult,
    MultimodalRetrievalResponse,
)
from app.services.rag_service import RAGPipelineService, build_rag_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fused_result(
    rank: int,
    evidence_id: str,
    answer: str = "Severe structural damage",
    rrf_score: float = 0.9,
) -> FusedEvidenceResult:
    return FusedEvidenceResult(
        rank=rank,
        rrf_score=rrf_score,
        evidence_id=evidence_id,
        record_id=f"rec_{evidence_id}",
        task_type="Damage Assessment",
        source_type="qa",
        text=f"Text for {evidence_id}",
        question="How severe is the damage?",
        answer=answer,
        disaster_type="Earthquake",
        location="Turkey",
    )


def make_mm_response(
    fused_results: List[FusedEvidenceResult],
    fusion_mode: str = "text_only",
) -> MultimodalRetrievalResponse:
    return MultimodalRetrievalResponse(
        question="How severe is the damage?",
        task_type="Damage Assessment",
        image_path=None,
        fusion_mode=fusion_mode,
        text_results_count=len(fused_results),
        image_results_count=0,
        fused_results=fused_results,
        latency_ms=5.0,
    )


def make_vlm_response(prediction: str = "Severe damage", status: str = "SUCCESS") -> Any:
    mock = MagicMock()
    mock.prediction = prediction
    mock.status = status
    mock.error_message = None
    mock.model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    mock.device = "cpu"
    return mock


def make_rag_service(
    fused_results: List[FusedEvidenceResult],
    fusion_mode: str = "text_only",
    vlm_prediction: str = "The earthquake caused severe structural damage.",
    vlm_status: str = "SUCCESS",
) -> RAGPipelineService:
    mm_mock = MagicMock()
    mm_mock.retrieve.return_value = make_mm_response(fused_results, fusion_mode)

    vlm_mock = MagicMock()
    vlm_mock.generate.return_value = make_vlm_response(vlm_prediction, vlm_status)

    return RAGPipelineService(
        retrieval_service=mm_mock,
        vlm_service=vlm_mock,
    )


# ---------------------------------------------------------------------------
# Context Builder Tests
# ---------------------------------------------------------------------------


class TestBuildRagContext:

    def test_context_contains_evidence(self):
        """Assembled context includes evidence from fused results."""
        fused = [make_fused_result(1, "evi_001")]
        ctx = build_rag_context("What damage occurred?", "Damage Assessment", fused)

        assert isinstance(ctx, RAGContext)
        assert "Evidence 1" in ctx.evidence_block
        assert "How severe is the damage?" in ctx.evidence_block
        assert "Severe structural damage" in ctx.evidence_block

    def test_full_prompt_contains_question(self):
        """Full prompt includes the question and evidence."""
        fused = [make_fused_result(1, "evi_001")]
        ctx = build_rag_context("Is there flood damage?", None, fused)

        assert "Is there flood damage?" in ctx.full_prompt
        assert "Evidence 1" in ctx.full_prompt

    def test_char_budget_enforced(self):
        """Context assembly stops when max_context_chars is reached."""
        # Create many results to overflow budget
        fused = [make_fused_result(i, f"evi_{i:03d}") for i in range(1, 50)]
        ctx = build_rag_context("question", None, fused, max_context_chars=500)

        assert ctx.context_chars <= 600  # Allow small overshoot for last snippet

    def test_evidence_count_matches_snippets(self):
        """evidence_count matches the number of EvidenceSnippet objects."""
        fused = [make_fused_result(i, f"evi_{i:03d}") for i in range(1, 4)]
        ctx = build_rag_context("damage?", None, fused, max_context_chars=10000)

        assert ctx.evidence_count == len(ctx.evidence_snippets)
        assert ctx.evidence_count == 3

    def test_empty_fused_results_returns_empty_context(self):
        """No evidence → empty evidence block but valid RAGContext."""
        ctx = build_rag_context("question", None, [], max_context_chars=1000)
        assert ctx.evidence_count == 0
        assert ctx.evidence_snippets == []

    def test_task_type_included_in_prompt(self):
        """Task type appears in the full prompt when provided."""
        fused = [make_fused_result(1, "evi_001")]
        ctx = build_rag_context("?", "Flood Detection", fused)

        assert "Flood Detection" in ctx.full_prompt

    def test_snippets_have_correct_rrf_score(self):
        """EvidenceSnippet carries the correct rrf_score from fused results."""
        fused = [make_fused_result(1, "evi_001", rrf_score=0.42)]
        ctx = build_rag_context("q", None, fused)

        assert ctx.evidence_snippets[0].rrf_score == pytest.approx(0.42, abs=1e-6)


# ---------------------------------------------------------------------------
# RAGPipelineService Tests
# ---------------------------------------------------------------------------


class TestRAGPipelineService:

    def test_successful_pipeline_returns_prediction(self):
        """Full pipeline produces a RAGResponse with SUCCESS status."""
        fused = [make_fused_result(1, "evi_001"), make_fused_result(2, "evi_002")]
        service = make_rag_service(fused, vlm_prediction="Collapsed buildings observed.")

        request = RAGRequest(question="What damage occurred?", task_type="Damage Assessment")
        response = service.run(request)

        assert isinstance(response, RAGResponse)
        assert response.status == "SUCCESS"
        assert response.prediction == "Collapsed buildings observed."
        assert response.evidence_count >= 1

    def test_no_evidence_returns_no_evidence_status(self):
        """Pipeline returns NO_EVIDENCE status when retrieval returns nothing."""
        service = make_rag_service([])

        request = RAGRequest(question="obscure unindexed question")
        response = service.run(request)

        assert response.status == "NO_EVIDENCE"
        assert response.prediction == ""
        assert response.evidence_count == 0

    def test_latency_fields_populated(self):
        """Response always contains non-negative latency values."""
        fused = [make_fused_result(1, "evi_001")]
        service = make_rag_service(fused)

        request = RAGRequest(question="flood damage?")
        response = service.run(request)

        assert response.retrieval_latency_ms >= 0.0
        assert response.vlm_latency_ms >= 0.0
        assert response.total_latency_ms >= 0.0
        assert response.total_latency_ms >= response.retrieval_latency_ms

    def test_provenance_included_when_requested(self):
        """When include_provenance=True, evidence_snippets is populated."""
        fused = [make_fused_result(1, "evi_001"), make_fused_result(2, "evi_002")]
        service = make_rag_service(fused)

        request = RAGRequest(question="What happened?", include_provenance=True)
        response = service.run(request)

        assert len(response.evidence_snippets) > 0
        assert all(isinstance(s, EvidenceSnippet) for s in response.evidence_snippets)

    def test_provenance_excluded_when_not_requested(self):
        """When include_provenance=False, evidence_snippets is empty."""
        fused = [make_fused_result(1, "evi_001")]
        service = make_rag_service(fused)

        request = RAGRequest(question="What happened?", include_provenance=False)
        response = service.run(request)

        assert response.evidence_snippets == []

    def test_vlm_error_propagated(self):
        """VLM ERROR status is forwarded in RAGResponse."""
        fused = [make_fused_result(1, "evi_001")]

        mm_mock = MagicMock()
        mm_mock.retrieve.return_value = make_mm_response(fused)

        vlm_mock = MagicMock()
        err_response = make_vlm_response("", "ERROR")
        err_response.error_message = "Model OOM"
        vlm_mock.generate.return_value = err_response

        service = RAGPipelineService(
            retrieval_service=mm_mock,
            vlm_service=vlm_mock,
        )

        request = RAGRequest(question="What damage occurred?")
        response = service.run(request)

        assert response.status == "ERROR"
        assert "OOM" in (response.error_message or "")

    def test_retrieval_mode_propagated(self):
        """Retrieval mode from multimodal service is propagated to response."""
        fused = [make_fused_result(1, "evi_001")]
        service = make_rag_service(fused, fusion_mode="multimodal_rrf")

        request = RAGRequest(question="damage?")
        response = service.run(request)

        assert response.retrieval_mode == "multimodal_rrf"

    def test_vlm_receives_augmented_prompt(self):
        """VLM generate() is called with a VLMInferenceRequest."""
        from app.schemas.vlm import VLMInferenceRequest

        fused = [make_fused_result(1, "evi_001")]

        mm_mock = MagicMock()
        mm_mock.retrieve.return_value = make_mm_response(fused)

        vlm_mock = MagicMock()
        vlm_mock.generate.return_value = make_vlm_response("answer")

        service = RAGPipelineService(
            retrieval_service=mm_mock,
            vlm_service=vlm_mock,
        )

        request = RAGRequest(question="How severe is the damage?")
        service.run(request)

        vlm_mock.generate.assert_called_once()
        call_arg = vlm_mock.generate.call_args[0][0]
        assert isinstance(call_arg, VLMInferenceRequest)
        # Context evidence should be injected into the VLM question
        assert "HISTORICAL EVIDENCE" in call_arg.question
