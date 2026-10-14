"""
Schemas for Live Disaster Evidence Ingestion (Topic 09).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.evidence import EvidenceRecord


class LiveDisasterFeedItem(BaseModel):
    """Normalized incident item fetched from a live disaster feed."""
    event_id: str
    title: str
    disaster_type: str
    severity: str = "MODERATE"
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timestamp: str
    url: Optional[str] = None
    summary: str
    source_feed: str  # "usgs", "reliefweb", "gdacs", "custom"


class LiveFeedIngestRequest(BaseModel):
    """Request payload to fetch and ingest live disaster evidence."""
    feed_source: str = Field(default="usgs", description="Feed provider ('usgs', 'reliefweb', 'mock')")
    min_magnitude: float = Field(default=4.5, ge=1.0, le=10.0, description="Minimum earthquake magnitude for USGS")
    limit: int = Field(default=10, ge=1, le=100, description="Max items to fetch")
    auto_index: bool = Field(default=False, description="Whether to index directly into Qdrant")


class LiveFeedIngestResponse(BaseModel):
    """Outcome of live disaster evidence ingestion."""
    feed_source: str
    fetched_count: int
    converted_evidence_count: int
    indexed_count: int = 0
    items: List[LiveDisasterFeedItem] = Field(default_factory=list)
    latency_ms: float = 0.0
    status: str = "SUCCESS"
