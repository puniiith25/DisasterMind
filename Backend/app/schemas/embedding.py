"""Strongly-typed schemas for text embeddings and embedding cache."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class EmbeddingRepresentation(str, Enum):
    """Supported embedding representations for BGE-M3."""
    DENSE = "dense"
    SPARSE = "sparse"
    MULTI_VECTOR = "multi_vector"


class EmbeddingMetadata(BaseModel):
    """Metadata regarding embedding model and configuration."""
    model_id: str = "BAAI/bge-m3"
    dimension: int = 1024
    representation: EmbeddingRepresentation = EmbeddingRepresentation.DENSE
    normalized: bool = True
    device: str = "mps"


class EmbeddingBatchResult(BaseModel):
    """Result of batch embedding computation."""
    count: int
    cached_count: int
    computed_count: int
    latency_ms: float
    metadata: EmbeddingMetadata


class EmbeddingBenchmarkReport(BaseModel):
    """Report summarizing embedding runtime, throughput, and memory measurements."""
    timestamp: str
    model_id: str
    device: str
    dimension: int
    representation: str
    total_texts: int
    batch_size: int
    total_latency_seconds: float
    throughput_items_per_sec: float
    cache_hit_rate: float
    memory_mb: Optional[float] = None
