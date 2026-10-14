"""
Schemas for Evidence Verification and Contradiction Detection (Topic 10).
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class VerificationVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED = "UNVERIFIED"


class ClaimVerificationResult(BaseModel):
    """Result of checking a single claim against retrieved evidence."""
    claim: str
    verdict: VerificationVerdict
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_evidence_id: Optional[str] = None
    explanation: str


class VerificationReport(BaseModel):
    """Comprehensive fact-checking report for a disaster assessment output."""
    total_claims: int
    supported_count: int
    contradicted_count: int
    unverified_count: int
    veracity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    has_critical_contradiction: bool = False
    claim_results: List[ClaimVerificationResult] = Field(default_factory=list)
