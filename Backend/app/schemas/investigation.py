"""
DisasterMind Investigation Schemas (Topic 14).

Provides strongly-typed, validated Pydantic models for:
- Image EXIF metadata (GPS, timestamp, provenance)
- Normalized disaster alerts (USGS, GDACS, Weather, Road hazards)
- Evidence citations with clickable URLs and timestamps
- 10-point evidence-grounded investigation answers
- Route & travel risk investigation payloads
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ImageMetadataInfo(BaseModel):
    """Trustworthy metadata extracted from uploaded disaster images."""
    has_gps: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    capture_time: Optional[str] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    location_provenance: str = Field(
        default="absent",
        description="'image_exif_metadata', 'user_confirmed', 'geocoded_text', or 'absent'"
    )
    needs_location_confirmation: bool = True
    notes: Optional[str] = None


class NormalizedDisasterAlert(BaseModel):
    """Unified internal representation of external official disaster bulletins."""
    alert_id: str
    source: str = Field(..., description="Provider name: e.g. 'USGS', 'GDACS', 'National Weather Service', 'Open-Meteo'")
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    event_type: str = Field(..., description="'Earthquake', 'Flood', 'Tropical Cyclone', 'Severe Storm', 'Wildfire', 'Tsunami', etc.")
    title: str
    severity: str = Field(default="UNKNOWN", description="'CRITICAL', 'MAJOR', 'MODERATE', 'ADVISORY', 'UNKNOWN'")
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    event_location: str
    distance_km: Optional[float] = None
    affects_location: bool = Field(
        default=False,
        description="True if the event specifically impacted the target location, False if only nearby"
    )
    event_time: Optional[str] = None
    published_at: Optional[str] = None
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    summary: str
    limitations: Optional[str] = None


class EvidenceCitation(BaseModel):
    """Citation referencing verifiable historical or live evidence."""
    id: str
    title: str
    source: str
    source_url: Optional[str] = None
    is_historical: bool = True
    observed_at: Optional[str] = None
    published_at: Optional[str] = None
    retrieved_at: Optional[str] = None
    snippet: str
    distance_km: Optional[float] = None


class RecentImageryStatus(BaseModel):
    """Transparent disclosure of recent satellite/aerial imagery availability."""
    available: bool = False
    provider: Optional[str] = None
    capture_date: Optional[str] = None
    retrieval_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tile_url: Optional[str] = None
    limitation_reason: str = (
        "Commercial high-resolution post-disaster satellite imagery requires dedicated tasking "
        "and is not freely accessible in real time. Historical DisasterM3 baseline imagery is used for context."
    )


class SourceCheckedStatus(BaseModel):
    """Status record of external official source queried during investigation."""
    provider: str
    status: str = Field(..., description="'SUCCESS', 'NO_RECORDS_FOUND', 'UNAVAILABLE', 'FAILED'")
    records_count: int = 0
    latency_ms: float = 0.0
    details: Optional[str] = None


class InvestigationGroundedAnswer(BaseModel):
    """
    Standardized 10-Point Evidence-Grounded Disaster Intelligence Response.
    Never invents unvalidated numerical confidence percentages.
    """
    direct_answer: str
    historical_context: Optional[str] = None
    latest_official_alert: Optional[NormalizedDisasterAlert] = None
    supporting_evidence: List[EvidenceCitation] = Field(default_factory=list)
    source_names_and_urls: List[Dict[str, str]] = Field(default_factory=list)
    timestamps: Dict[str, Optional[str]] = Field(default_factory=dict)
    confirmed_location: Optional[str] = None
    location_provenance: str = "unspecified"
    location_coords: Optional[Dict[str, float]] = None
    conflicting_evidence: Optional[str] = None
    missing_information_and_uncertainty: str
    practical_next_verification_step: str
    recent_imagery: RecentImageryStatus = Field(default_factory=RecentImageryStatus)
    sources_checked: List[SourceCheckedStatus] = Field(default_factory=list)


class LocationInvestigationRequest(BaseModel):
    """Payload for location-based disaster assessment."""
    location: str = Field(..., min_length=2, description="City, region, or place name")
    question: Optional[str] = Field(
        default="What is the current disaster and hazard situation in this area?",
        description="User question"
    )
    confirmed_latitude: Optional[float] = None
    confirmed_longitude: Optional[float] = None
    target_date: Optional[str] = Field(
        default=None,
        description="Intended travel date or date of inquiry (e.g. '2026-10-24', 'tomorrow')"
    )


class RouteHazardItem(BaseModel):
    """Identified danger, closure, or weather alert along a transit corridor."""
    hazard_type: str
    title: str
    severity: str
    location_description: str
    latitude: float
    longitude: float
    distance_to_route_km: float
    source: str
    source_url: Optional[str] = None
    reported_at: Optional[str] = None
    status: str = Field(
        default="REPORTED_HAZARD",
        description="'CONFIRMED_CLOSURE', 'REPORTED_HAZARD', 'WEATHER_WARNING', 'UNVERIFIED_CONDITION'"
    )
    details: str


class RouteInvestigationRequest(BaseModel):
    """Payload for route and road risk investigation."""
    origin: str = Field(..., min_length=2, description="Departure location or address")
    destination: str = Field(..., min_length=2, description="Destination location or address")
    departure_time: Optional[str] = Field(
        default=None,
        description="ISO timestamp or date string of departure (e.g. '2026-10-30')"
    )
    question: Optional[str] = Field(
        default="Are there any road closures, floods, or severe hazards along this route?",
        description="Specific question regarding transit corridor"
    )


class RouteInvestigationResponse(BaseModel):
    """Complete route risk investigation output."""
    origin: str
    destination: str
    origin_coords: Optional[Dict[str, float]] = None
    destination_coords: Optional[Dict[str, float]] = None
    departure_time: Optional[str] = None
    is_forward_looking: bool = False
    forward_looking_warning: Optional[str] = None
    route_found: bool = False
    route_geometry: Optional[Dict[str, Any]] = None  # GeoJSON LineString coordinates
    distance_km: Optional[float] = None
    duration_minutes: Optional[float] = None
    alternative_routes: List[Dict[str, Any]] = Field(default_factory=list)
    detected_hazards: List[RouteHazardItem] = Field(default_factory=list)
    hazard_summary: str
    safety_advisory: str
    unverified_stretches: str
    sources_checked: List[SourceCheckedStatus] = Field(default_factory=list)
    evidence_citations: List[EvidenceCitation] = Field(default_factory=list)
    grounded_answer: InvestigationGroundedAnswer
