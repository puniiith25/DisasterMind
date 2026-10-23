"""
Unit tests for Live Incident Intelligence and Image Analysis Service.
"""

from __future__ import annotations

import pytest
from PIL import Image

from app.schemas.intelligence import IntelligenceResponse
from app.services.intelligence_service import (
    IncidentIntelligenceService,
    get_intelligence_service,
)


from unittest.mock import MagicMock
from app.schemas.vlm import VLMInferenceResponse


@pytest.fixture
def mock_vlm():
    mock = MagicMock()
    mock.generate.return_value = VLMInferenceResponse(
        prediction=(
            "WHAT: Severe earthquake structural collapse observed across buildings.\n"
            "WHERE: Antakya, Turkey disaster sector.\n"
            "HOW: Triggered by strike-slip fault rupture exceeding shear strength.\n"
            "IMPACT: Massive building collapse and emergency priority."
        ),
        device="mps",
        tokens_generated=32,
        latency_ms=45.2,
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    return mock


@pytest.fixture
def intelligence_service(mock_vlm):
    return IncidentIntelligenceService(vlm_service=mock_vlm)


def test_detect_disaster_type(intelligence_service):
    assert intelligence_service.detect_disaster_type("Severe earthquake building collapse") == "Earthquake"
    assert intelligence_service.detect_disaster_type("Rivers overflowing and submerged roads") == "Flood"
    assert intelligence_service.detect_disaster_type("Flames spreading across forest canopy") == "Wildfire"


def test_analyze_incident_offline(intelligence_service, tmp_path):
    # Create test image
    img_path = tmp_path / "damage_scene.png"
    img = Image.new("RGB", (100, 100), color=(150, 75, 50))
    img.save(img_path)

    response = intelligence_service.analyze_incident(
        question="What happened in Antakya and what caused the damage?",
        image_path=str(img_path),
        search_live=False,  # Use resilient local templates
    )

    assert isinstance(response, IntelligenceResponse)
    assert response.status == "SUCCESS"
    assert response.image_analyzed is True
    assert "Earthquake" in response.breakdown.detected_disaster_type
    assert "Antakya" in response.breakdown.where_it_happened
    assert len(response.breakdown.how_it_happened) > 10
    assert len(response.breakdown.what_happened) > 10
    assert len(response.source_provenance) >= 2


def test_singleton_intelligence_service():
    s1 = get_intelligence_service()
    s2 = get_intelligence_service()
    assert s1 is s2


def test_non_disaster_analysis_no_canned_text(mock_vlm, tmp_path):
    # Mock VLM describing a movie poster / non-disaster scene
    mock_vlm.generate.return_value = VLMInferenceResponse(
        prediction=(
            "DISASTER TYPE: Non-Disaster / Normal Scene\n"
            "CONFIDENCE: 0.98\n"
            "WHAT: The image depicts a promotional movie poster featuring cast members and artistic graphics.\n"
            "WHERE: Graphic design / promotional media.\n"
            "HOW: None / Not Applicable\n"
            "IMPACT: No disaster hazards observed. The image presents no emergency response priorities."
        ),
        device="mps",
        tokens_generated=45,
        latency_ms=30.0,
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    svc = IncidentIntelligenceService(vlm_service=mock_vlm)
    poster_path = tmp_path / "poster.png"
    Image.new("RGB", (100, 100), color=(200, 200, 255)).save(poster_path)

    response = svc.analyze_incident(
        question="What is this image?",
        image_path=str(poster_path),
        search_live=True,
    )

    assert response.status == "SUCCESS"
    assert response.breakdown.detected_disaster_type == "Non-Disaster / Normal Scene"
    assert "promotional movie poster" in response.breakdown.what_happened
    assert "pancake" not in response.breakdown.how_it_happened.lower()
    assert "ductility" not in response.breakdown.how_it_happened.lower()
    # Live seismic feeds should NOT be attached to non-disaster scenes
    assert len(response.live_internet_evidence) == 0

