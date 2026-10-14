"""
Unit tests for Topic 08: Grounded Generation & Citation Engine.
"""

from __future__ import annotations

import pytest

from app.schemas.grounding import GroundedPrediction, HallucinationRisk
from app.schemas.rag import EvidenceSnippet
from app.services.grounding_service import GroundingService, get_grounding_service


@pytest.fixture
def grounding_service():
    return GroundingService(support_threshold=0.25)


@pytest.fixture
def mock_evidence():
    return [
        EvidenceSnippet(
            rank=1,
            evidence_id="evi_001",
            task_type="Building Damage Counting",
            source_type="qa",
            question="How many buildings collapsed after the earthquake in Antakya?",
            answer="There are 45 collapsed buildings and 12 damaged roads reported.",
            rrf_score=0.95,
        ),
        EvidenceSnippet(
            rank=2,
            evidence_id="evi_002",
            task_type="Disaster Scene Recognition",
            source_type="qa",
            question="What type of flood event occurred in Derna, Libya?",
            answer="Catastrophic dam failure causing severe coastal flooding and debris accumulation.",
            rrf_score=0.82,
        ),
    ]


def test_claim_decomposition(grounding_service):
    text = "The earthquake caused severe destruction. 45 buildings collapsed in Antakya! Roads are blocked."
    claims = grounding_service.decompose_claims(text)
    assert len(claims) == 3
    assert "The earthquake caused severe destruction." in claims


def test_grounded_prediction_with_supported_claims(grounding_service, mock_evidence):
    pred_text = "In Antakya, 45 collapsed buildings were confirmed following the earthquake."
    result = grounding_service.evaluate(pred_text, mock_evidence)

    assert isinstance(result, GroundedPrediction)
    assert result.grounding_score > 0.4
    assert result.hallucination_risk in (HallucinationRisk.LOW, HallucinationRisk.MEDIUM)
    assert len(result.citations) >= 1
    assert result.citations[0].evidence_id == "evi_001"
    assert "evi_001" in result.grounded_prediction


def test_hallucination_detection_unsupported_claim(grounding_service, mock_evidence):
    pred_text = "A coronal mass ejection caused cosmic communication blackout across orbital satellites."
    result = grounding_service.evaluate(pred_text, mock_evidence)

    assert result.grounding_score == 0.0
    assert result.hallucination_risk == HallucinationRisk.HIGH
    assert len(result.citations) == 0



def test_empty_prediction_and_empty_evidence(grounding_service, mock_evidence):
    empty_result = grounding_service.evaluate("", mock_evidence)
    assert empty_result.grounding_score == 0.0

    no_evidence_result = grounding_service.evaluate("Some disaster occurred.", [])
    assert no_evidence_result.grounding_score == 0.0
    assert no_evidence_result.hallucination_risk == HallucinationRisk.HIGH


def test_singleton_grounding_service():
    svc1 = get_grounding_service()
    svc2 = get_grounding_service()
    assert svc1 is svc2
