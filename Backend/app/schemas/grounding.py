"""
Schemas for Grounded Generation and Citation Verification (Topic 08).
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.rag import EvidenceSnippet


class HallucinationRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Citation(BaseModel):
    """Citation linking a claim to a specific evidence snippet."""
    citation_id: str
    evidence_id: str
    rank: int
    task_type: str
    source_type: str
    snippet_preview: str


class GroundedClaim(BaseModel):
    """Individual statement extracted from model prediction and evaluated for grounding."""
    claim_text: str
    is_supported: bool
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    supporting_evidence_ids: List[str] = Field(default_factory=list)


class GroundedPrediction(BaseModel):
    """Complete grounded prediction with extracted citations and confidence audit."""
    original_prediction: str
    grounded_prediction: str
    citations: List[Citation] = Field(default_factory=list)
    claims: List[GroundedClaim] = Field(default_factory=list)
    grounding_score: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucination_risk: HallucinationRisk = HallucinationRisk.LOW
    evidence_count: int = 0
