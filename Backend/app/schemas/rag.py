"""
Schemas for the DisasterMind RAG Pipeline (DM-07).

Covers the full Retrieve-Augment-Generate cycle:
  - RAGRequest: full question + optional images + retrieval config
  - RAGContext: formatted evidence context passed to the VLM
  - RAGResponse: structured pipeline output with provenance
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    """Input to the full RAG pipeline."""
    question: str
    task_type: Optional[str] = None
    pre_image_path: Optional[str] = None
    post_image_path: Optional[str] = None
    image_path: Optional[str] = None
    text_top_k: int = Field(default=5, ge=1, le=20)
    image_top_k: int = Field(default=3, ge=0, le=10)
    fusion_top_k: int = Field(default=5, ge=1, le=20)
    max_context_chars: int = Field(default=3000, ge=100)
    max_new_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    include_provenance: bool = True


class EvidenceSnippet(BaseModel):
    """A single piece of evidence injected into the VLM context."""
    rank: int
    evidence_id: str
    task_type: str
    source_type: str
    question: str
    answer: str
    rrf_score: float
    disaster_type: Optional[str] = None
    location: Optional[str] = None


class RAGContext(BaseModel):
    """Formatted evidence context assembled for the VLM prompt."""
    system_prompt: str
    evidence_block: str
    full_prompt: str
    evidence_count: int
    evidence_snippets: List[EvidenceSnippet] = Field(default_factory=list)
    context_chars: int = 0
    retrieval_mode: str = "text_only"


class RAGResponse(BaseModel):
    """Complete RAG pipeline response with prediction and provenance."""
    question: str
    task_type: Optional[str]
    prediction: str
    status: str  # "SUCCESS" | "ERROR" | "NO_EVIDENCE"
    error_message: Optional[str] = None

    # Retrieval provenance
    retrieval_mode: str
    evidence_count: int
    evidence_snippets: List[EvidenceSnippet] = Field(default_factory=list)
    context_chars: int = 0

    # Timing
    retrieval_latency_ms: float = 0.0
    vlm_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Model info
    vlm_model_id: str = ""
    device: str = ""
