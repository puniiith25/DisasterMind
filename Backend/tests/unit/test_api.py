"""
Unit tests for DisasterMind FastAPI Server (DM-08).

Verifies all online inference and retrieval endpoints:
- GET /
- GET /health
- POST /api/v1/rag/predict
- POST /api/v1/retrieval/multimodal
- POST /api/v1/retrieval/dense
- POST /api/v1/vlm/generate
- GET /api/v1/dataset/stats
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from starlette.testclient import TestClient

from app.main import (
    app,
    get_multimodal_retrieval,
    get_qdrant_index,
    get_rag_pipeline,
    get_vlm,
)
from app.schemas.rag import EvidenceSnippet, RAGResponse
from app.schemas.retrieval import RetrievalResponse, RetrievalResult
from app.schemas.vlm import VLMInferenceResponse
from app.services.multimodal_service import (
    FusedEvidenceResult,
    MultimodalRetrievalResponse,
)


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Test Root & Health
# ---------------------------------------------------------------------------


def test_root_endpoint(client):
    """GET / should return 200 and service metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "DisasterMind API Server Running"
    assert "endpoints" in data
    assert data["endpoints"]["rag"] == "/api/v1/rag/predict"


def test_health_endpoint_healthy(client):
    """GET /health should report collection status."""
    mock_qdrant = MagicMock()
    mock_qdrant.collection_info.return_value = {
        "exists": True,
        "points_count": 1263,
    }

    app.dependency_overrides[get_qdrant_index] = lambda: mock_qdrant
    try:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "HEALTHY"
        assert data["qdrant_indexed"] is True
        assert data["qdrant_points"] == 1263
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Test RAG Endpoint
# ---------------------------------------------------------------------------


def test_rag_predict_endpoint_success(client):
    """POST /api/v1/rag/predict returns full RAG prediction with evidence."""
    mock_rag = MagicMock()
    mock_rag.run.return_value = RAGResponse(
        question="How many buildings are damaged?",
        task_type="Building Damage Counting",
        prediction="There are 12 collapsed buildings.",
        status="SUCCESS",
        retrieval_mode="multimodal_rrf",
        evidence_count=1,
        evidence_snippets=[
            EvidenceSnippet(
                rank=1,
                evidence_id="evi_001",
                task_type="Building Damage Counting",
                source_type="qa",
                question="How many buildings are damaged?",
                answer="12 collapsed buildings",
                rrf_score=0.016,
            )
        ],
        context_chars=120,
        retrieval_latency_ms=12.5,
        vlm_latency_ms=45.0,
        total_latency_ms=57.5,
        vlm_model_id="Qwen2.5-VL-3B-Instruct",
        device="cpu",
    )

    app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag
    try:
        payload = {
            "question": "How many buildings are damaged?",
            "task_type": "Building Damage Counting",
            "text_top_k": 3,
            "fusion_top_k": 3,
        }
        response = client.post("/api/v1/rag/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["prediction"] == "There are 12 collapsed buildings."
        assert len(data["evidence_snippets"]) == 1
        assert data["retrieval_mode"] == "multimodal_rrf"
    finally:
        app.dependency_overrides.clear()


def test_rag_predict_endpoint_error_handling(client):
    """POST /api/v1/rag/predict handles service exceptions cleanly."""
    mock_rag = MagicMock()
    mock_rag.run.side_effect = RuntimeError("VLM Model OOM")

    app.dependency_overrides[get_rag_pipeline] = lambda: mock_rag
    try:
        payload = {"question": "Test query"}
        response = client.post("/api/v1/rag/predict", json=payload)
        assert response.status_code == 500
        assert "VLM Model OOM" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Test Retrieval Endpoints
# ---------------------------------------------------------------------------


def test_retrieval_multimodal_endpoint(client):
    """POST /api/v1/retrieval/multimodal returns fused evidence results."""
    mock_mm = MagicMock()
    mock_mm.retrieve.return_value = MultimodalRetrievalResponse(
        question="What disaster occurred?",
        task_type="Disaster Type Recognition",
        image_path=None,
        fusion_mode="text_only",
        text_results_count=1,
        image_results_count=0,
        fused_results=[
            FusedEvidenceResult(
                rank=1,
                rrf_score=0.016,
                evidence_id="evi_100",
                record_id="dm3_100",
                task_type="Disaster Type Recognition",
                source_type="qa",
                text="Earthquake in Turkey",
                question="What disaster?",
                answer="Earthquake",
            )
        ],
        latency_ms=8.0,
    )

    app.dependency_overrides[get_multimodal_retrieval] = lambda: mock_mm
    try:
        payload = {"question": "What disaster occurred?"}
        response = client.post("/api/v1/retrieval/multimodal", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["fusion_mode"] == "text_only"
        assert len(data["fused_results"]) == 1
    finally:
        app.dependency_overrides.clear()


def test_retrieval_dense_endpoint(client):
    """POST /api/v1/retrieval/dense executes vector similarity query."""
    mock_qdrant = MagicMock()
    mock_qdrant.retrieve.return_value = RetrievalResponse(
        query="Flood extent",
        top_k=2,
        collection_name="disasterm3_text",
        results=[
            RetrievalResult(
                rank=1,
                score=0.92,
                evidence_id="evi_200",
                record_id="dm3_200",
                task_type="Damage Assessment",
                source_type="qa",
                text="Severe flooding in coastal areas",
                question="Flood damage?",
                answer="Severe flooding",
            )
        ],
        latency_ms=4.5,
    )

    app.dependency_overrides[get_qdrant_index] = lambda: mock_qdrant
    try:
        payload = {"query": "Flood extent", "top_k": 2}
        response = client.post("/api/v1/retrieval/dense", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["collection_name"] == "disasterm3_text"
        assert len(data["results"]) == 1
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Test VLM Baseline Endpoint
# ---------------------------------------------------------------------------


def test_vlm_generate_endpoint(client):
    """POST /api/v1/vlm/generate executes unaugmented VLM generation."""
    mock_vlm = MagicMock()
    mock_vlm.generate.return_value = VLMInferenceResponse(
        prompt="Describe the scene",
        prediction="An aerial view showing flood inundation.",
        latency_ms=30.0,
        device="cpu",
        model_id="Qwen2.5-VL-3B-Instruct",
        status="SUCCESS",
    )

    app.dependency_overrides[get_vlm] = lambda: mock_vlm
    try:
        payload = {"question": "Describe the scene"}
        response = client.post("/api/v1/vlm/generate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["prediction"] == "An aerial view showing flood inundation."
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Test Dataset Stats Endpoint
# ---------------------------------------------------------------------------


def test_dataset_stats_endpoint(client):
    """GET /api/v1/dataset/stats returns json reports."""
    response = client.get("/api/v1/dataset/stats")
    assert response.status_code == 200
    data = response.json()
    assert "validation_report" in data
    assert "corpus_statistics" in data


# ---------------------------------------------------------------------------
# Test Grounding, Live Feed, and Verification Endpoints
# ---------------------------------------------------------------------------


def test_grounding_evaluate_endpoint(client):
    """POST /api/v1/grounding/evaluate evaluates prediction and citations."""
    from app.main import get_grounding
    from app.schemas.grounding import GroundedPrediction, HallucinationRisk

    mock_grounding = MagicMock()
    mock_grounding.evaluate.return_value = GroundedPrediction(
        original_prediction="45 buildings collapsed in the earthquake.",
        grounded_prediction="45 buildings collapsed in the earthquake. [cit_1: evi_001]",
        citations=[],
        claims=[],
        grounding_score=0.88,
        hallucination_risk=HallucinationRisk.LOW,
        evidence_count=1,
    )

    app.dependency_overrides[get_grounding] = lambda: mock_grounding
    try:
        payload = {
            "prediction": "45 buildings collapsed in the earthquake.",
            "evidence_snippets": [
                {
                    "rank": 1,
                    "evidence_id": "evi_001",
                    "task_type": "Building Damage Counting",
                    "source_type": "qa",
                    "question": "Buildings collapsed?",
                    "answer": "45 buildings collapsed",
                    "rrf_score": 0.95,
                }
            ],
        }
        response = client.post("/api/v1/grounding/evaluate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["grounding_score"] == 0.88
        assert data["hallucination_risk"] == "LOW"
    finally:
        app.dependency_overrides.clear()


def test_live_ingest_endpoint(client):
    """POST /api/v1/live/ingest fetches and normalizes live disaster bulletins."""
    from app.main import get_live_feed
    from app.schemas.live_feed import LiveFeedIngestResponse

    mock_live = MagicMock()
    mock_live.ingest.return_value = LiveFeedIngestResponse(
        feed_source="usgs",
        fetched_count=2,
        converted_evidence_count=2,
        indexed_count=0,
        items=[],
        latency_ms=15.0,
        status="SUCCESS",
    )

    app.dependency_overrides[get_live_feed] = lambda: mock_live
    try:
        payload = {"feed_source": "usgs", "limit": 2, "auto_index": False}
        response = client.post("/api/v1/live/ingest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["fetched_count"] == 2
    finally:
        app.dependency_overrides.clear()


def test_verification_verify_endpoint(client):
    """POST /api/v1/verification/verify executes fact-checking against evidence."""
    from app.main import get_verification
    from app.schemas.verification import ClaimVerificationResult, VerificationReport, VerificationVerdict

    mock_verify = MagicMock()
    mock_verify.verify.return_value = VerificationReport(
        total_claims=1,
        supported_count=1,
        contradicted_count=0,
        unverified_count=0,
        veracity_score=1.0,
        has_critical_contradiction=False,
        claim_results=[
            ClaimVerificationResult(
                claim="24 buildings collapsed.",
                verdict=VerificationVerdict.SUPPORTED,
                confidence=0.92,
                explanation="Fully confirmed by satellite record.",
            )
        ],
    )

    app.dependency_overrides[get_verification] = lambda: mock_verify
    try:
        payload = {
            "prediction": "24 buildings collapsed.",
            "evidence_snippets": [],
        }
        response = client.post("/api/v1/verification/verify", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["veracity_score"] == 1.0
        assert data["supported_count"] == 1
        assert data["has_critical_contradiction"] is False
    finally:
        app.dependency_overrides.clear()


def test_intelligence_analyze_json_endpoint(client):
    """POST /api/v1/intelligence/analyze returns What, Where, How, and live evidence."""
    from app.main import get_intelligence
    from app.schemas.intelligence import IncidentBreakdown, IntelligenceResponse

    mock_intel = MagicMock()
    mock_intel.analyze_incident.return_value = IntelligenceResponse(
        question="What disaster happened in Antakya?",
        summary="WHAT: Earthquake | WHERE: Antakya | HOW: Fault rupture",
        breakdown=IncidentBreakdown(
            what_happened="Catastrophic earthquake with 45 collapsed buildings.",
            where_it_happened="Antakya, Turkey",
            how_it_happened="Tectonic plate fault rupture.",
            impact_and_severity="Severe structural damage.",
            detected_disaster_type="Earthquake",
            confidence=0.95,
        ),
        live_internet_evidence=[],
        source_provenance=[{"source": "USGS", "details": "Live feed"}],
        latency_ms=12.0,
    )

    app.dependency_overrides[get_intelligence] = lambda: mock_intel
    try:
        payload = {"question": "What disaster happened in Antakya?"}
        response = client.post("/api/v1/intelligence/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["breakdown"]["detected_disaster_type"] == "Earthquake"
        assert "Antakya" in data["breakdown"]["where_it_happened"]
    finally:
        app.dependency_overrides.clear()


