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


@pytest.fixture
def intelligence_service():
    return IncidentIntelligenceService()


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
