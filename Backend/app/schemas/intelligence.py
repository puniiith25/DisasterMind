"""
Schemas for Live Incident Intelligence and Image Analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GeolocationMatch(BaseModel):
    """Guessed or verified location matched with visual clues and internet search."""
    place_name: str = Field(..., description="Guessed or confirmed city, region, and country")
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    confidence: float = 0.85
    provenance: str = "internet_visual_match"  # "exif_gps", "internet_visual_match", "visual_clues_speculation", "non_disaster_media"
    visual_clues: List[str] = Field(default_factory=list, description="Visual clues extracted from image: signage, architecture, etc.")
    matching_rationale: str = Field(default="", description="Reasoning connecting visual clues to internet records")
    internet_sources: List[Dict[str, str]] = Field(default_factory=list, description="Internet URLs and titles matching this location")


class IncidentBreakdown(BaseModel):
    """Detailed structural breakdown answering What, Where, How, and Severity."""
    what_happened: str = Field(..., description="Nature and scope of the disaster event")
    where_it_happened: str = Field(..., description="Geographic location, coordinates, or region")
    how_it_happened: str = Field(..., description="Mechanisms, triggers, or atmospheric/tectonic causes")
    impact_and_severity: str = Field(..., description="Damage metrics, casualties, structural compromise")
    detected_disaster_type: str = Field(..., description="Earthquake, Flood, Wildfire, Hurricane, etc.")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    inferred_location: Optional[str] = None
    geolocation_match: Optional[GeolocationMatch] = None


class LiveInternetEvidenceItem(BaseModel):
    """Live web/feed bulletin supporting the analysis."""
    source_name: str
    source_url: Optional[str] = None
    title: str
    timestamp: str
    snippet: str
    relevance_score: float = 0.9


class IntelligenceResponse(BaseModel):
    """Complete multimodal intelligence report for user uploaded image + query."""
    status: str = "SUCCESS"
    question: str
    image_analyzed: bool = True
    image_filename: Optional[str] = None
    image_filenames: List[str] = Field(default_factory=list)
    image_paths: List[str] = Field(default_factory=list, description="Server storage paths of analyzed images for follow-up turns")
    images_analyzed_count: int = 0
    summary: str
    breakdown: IncidentBreakdown
    geolocation_match: Optional[GeolocationMatch] = None
    live_internet_evidence: List[LiveInternetEvidenceItem] = Field(default_factory=list)
    source_provenance: List[Dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0

