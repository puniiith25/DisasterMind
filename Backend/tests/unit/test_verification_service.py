"""
Unit tests for Topic 10: Evidence Verification & Contradiction Detection.
"""

from __future__ import annotations

import pytest

from app.schemas.rag import EvidenceSnippet
from app.schemas.verification import VerificationVerdict
from app.services.verification_service import (
    EvidenceVerificationService,
    get_verification_service,
)


@pytest.fixture
def verification_service():
    return EvidenceVerificationService()


@pytest.fixture
def sample_evidence():
    return [
        EvidenceSnippet(
            rank=1,
            evidence_id="evi_earthquake_01",
            task_type="Building Damage Counting",
            source_type="qa",
            question="How many buildings collapsed after the earthquake?",
            answer="Satellite imagery shows exactly 24 collapsed buildings in the central square.",
            rrf_score=0.9,
            disaster_type="Earthquake",
            location="Izmir",
        )
    ]


def test_supported_claim_detection(verification_service, sample_evidence):
    claim = "There are 24 collapsed buildings confirmed in the central square after the earthquake."
    report = verification_service.verify(claim, sample_evidence)

    assert report.supported_count == 1
    assert report.contradicted_count == 0
    assert report.veracity_score > 0.5
    assert not report.has_critical_contradiction
    assert report.claim_results[0].verdict == VerificationVerdict.SUPPORTED


def test_numerical_contradiction_detection(verification_service, sample_evidence):
    claim = "Only 5 collapsed buildings were observed in the earthquake area."
    report = verification_service.verify(claim, sample_evidence)

    assert report.contradicted_count == 1
    assert report.has_critical_contradiction is True
    assert report.claim_results[0].verdict == VerificationVerdict.CONTRADICTED
    assert "Numerical contradiction" in report.claim_results[0].explanation


def test_disaster_type_contradiction(verification_service, sample_evidence):
    claim = "The hurricane destroyed several structures along the coast."
    report = verification_service.verify(claim, sample_evidence)

    assert report.contradicted_count == 1
    assert report.claim_results[0].verdict == VerificationVerdict.CONTRADICTED


def test_unverified_claim(verification_service, sample_evidence):
    claim = "The stock market opened higher this morning on economic growth signals."
    report = verification_service.verify(claim, sample_evidence)

    assert report.unverified_count == 1
    assert report.claim_results[0].verdict == VerificationVerdict.UNVERIFIED


def test_singleton_verification_service():
    v1 = get_verification_service()
    v2 = get_verification_service()
    assert v1 is v2
