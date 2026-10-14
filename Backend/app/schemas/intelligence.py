"""
Schemas for Live Incident Intelligence and Image Analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IncidentBreakdown(BaseModel):
    """Detailed structural breakdown answering What, Where, How, and Severity."""
    what_happened: str = Field(..., description="Nature and scope of the disaster event")
    where_it_happened: str = Field(..., description="Geographic location, coordinates, or region")
    how_it_happened: str = Field(..., description="Mechanisms, triggers, or atmospheric/tectonic causes")
    impact_and_severity: str = Field(..., description="Damage metrics, casualties, structural compromise")
    detected_disaster_type: str = Field(..., description="Earthquake, Flood, Wildfire, Hurricane, etc.")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)


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
    summary: str
    breakdown: IncidentBreakdown
    live_internet_evidence: List[LiveInternetEvidenceItem] = Field(default_factory=list)
    source_provenance: List[Dict[str, Any]] = Field(default_factory=list)
    latency_ms: float = 0.0
