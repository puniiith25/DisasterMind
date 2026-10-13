"""Strongly-typed schemas for Qdrant indexing and retrieval operations."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class CollectionStatus(str, Enum):
    """Qdrant collection lifecycle state."""
    CREATED = "created"
    UPDATED = "updated"
    EXISTS = "exists"
    DELETED = "deleted"


class IndexedPoint(BaseModel):
    """Lightweight record of a single indexed vector point."""
    model_config = ConfigDict(populate_by_name=True)

    point_id: int
    evidence_id: str
    record_id: str
    task_type: str
    source_type: str


class IndexBuildReport(BaseModel):
    """Summary of a Qdrant collection build / re-index operation."""
    collection_name: str
    vector_size: int
    distance_metric: str
    total_records: int
    indexed_count: int
    skipped_count: int
    status: CollectionStatus
    qdrant_path: str
    model_id: str
    latency_seconds: float
    throughput_items_per_sec: float
    cache_hit_rate: float
    indexed_points: List[IndexedPoint] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Single retrieval result from a Qdrant similarity search."""
    rank: int
    score: float
    evidence_id: str
    record_id: str
    task_type: str
    source_type: str
    text: str
    question: str
    answer: str
    disaster_type: Optional[str] = None
    location: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    """Response wrapping a ranked list of retrieval results."""
    query: str
    top_k: int
    collection_name: str
    results: List[RetrievalResult]
    latency_ms: float
