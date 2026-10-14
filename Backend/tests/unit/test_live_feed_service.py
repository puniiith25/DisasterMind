"""
Unit tests for Topic 09: Live Disaster Evidence Ingestion.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from app.schemas.evidence import EvidenceSourceType
from app.schemas.live_feed import LiveDisasterFeedItem, LiveFeedIngestRequest
from app.services.live_feed_service import LiveFeedService, get_live_feed_service


def test_fetch_usgs_offline_resilience():
    service = LiveFeedService(http_client=None)
    items = service.fetch_usgs_feed(limit=2)

    assert len(items) == 2
    assert all(isinstance(i, LiveDisasterFeedItem) for i in items)
    assert items[0].disaster_type == "Earthquake"
    assert items[0].source_feed == "usgs"


def test_fetch_usgs_with_mocked_http():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "features": [
            {
                "id": "mock_event_1",
                "properties": {
                    "title": "M 7.2 - Vanuatu Region",
                    "mag": 7.2,
                    "place": "Port-Vila, Vanuatu",
                    "time": 1700000000000,
                    "url": "https://earthquake.usgs.gov/mock",
                    "tsunami": 1,
                },
                "geometry": {"coordinates": [168.32, -17.74, 10.0]},
            }
        ]
    }
    mock_client.get.return_value = mock_resp

    service = LiveFeedService(http_client=mock_client)
    items = service.fetch_usgs_feed(min_magnitude=5.0, limit=5)

    assert len(items) == 1
    assert items[0].event_id == "usgs_mock_event_1"
    assert items[0].severity == "CRITICAL"
    assert items[0].latitude == -17.74
    assert items[0].longitude == 168.32


def test_convert_to_evidence_records():
    service = LiveFeedService()
    feed_items = [
        LiveDisasterFeedItem(
            event_id="test_01",
            title="M 6.1 - Crete, Greece",
            disaster_type="Earthquake",
            severity="MAJOR",
            location="Crete, Greece",
            timestamp="2026-10-14T12:00:00Z",
            summary="Strong earthquake felt across Aegean Sea.",
            source_feed="usgs",
        )
    ]

    records = service.convert_to_evidence(feed_items)
    assert len(records) == 1
    record = records[0]
    assert record.evidence_id.startswith("evi_live_")
    assert record.source_type == EvidenceSourceType.EXTERNAL_REPORT
    assert "Crete, Greece" in record.text
    assert record.metadata["is_live_ingested"] is True


def test_ingest_pipeline():
    service = LiveFeedService()
    req = LiveFeedIngestRequest(feed_source="usgs", limit=1, auto_index=False)
    resp = service.ingest(req)

    assert resp.status == "SUCCESS"
    assert resp.fetched_count == 1
    assert resp.converted_evidence_count == 1
    assert resp.latency_ms >= 0.0


def test_singleton_live_feed_service():
    s1 = get_live_feed_service()
    s2 = get_live_feed_service()
    assert s1 is s2
