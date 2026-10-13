"""Strongly-typed schemas for the multimodal evidence corpus."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvidenceSourceType(str, Enum):
    """Classification of evidence provenance."""
    DATASET_QA = "dataset_qa"
    VERIFIED_DESCRIPTION = "verified_description"
    EXTERNAL_REPORT = "external_report"
    MODEL_OUTPUT = "model_output"


class ImageRole(str, Enum):
    """Role of visual evidence in disaster context."""
    PRE_DISASTER = "pre_disaster"
    POST_DISASTER = "post_disaster"
    OBSERVATION = "observation"


class EvidenceImageReference(BaseModel):
    """Visual asset reference linked to an evidence item."""
    model_config = ConfigDict(str_strip_whitespace=True)

    image_path: str = Field(..., description="Path to image file")
    role: ImageRole = Field(..., description="Pre-disaster, post-disaster, or observation")
    modality: Optional[str] = Field(None, description="Modality e.g. Optical, SAR")


class EvidenceRecord(BaseModel):
    """Standardized multimodal evidence unit."""
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    evidence_id: str = Field(..., description="Deterministic stable identifier (e.g. evi_xxx)")
    record_id: Optional[str] = Field(None, description="Originating dataset record ID if applicable")
    text: str = Field(..., min_length=1, description="Primary textual evidence content")
    task_type: Optional[str] = Field(None, description="Disaster task category if applicable")
    question: Optional[str] = Field(None, description="Original question if this is a QA item")
    answer: Optional[str] = Field(None, description="Answer or ground truth if this is a QA item")
    images: List[EvidenceImageReference] = Field(default_factory=list, description="Associated image evidence")
    disaster_type: Optional[str] = Field(None, description="Disaster category (e.g. Earthquake, Flood, Tornado)")
    location: Optional[str] = Field(None, description="Geographic location if known; None if unverified")
    source_type: EvidenceSourceType = Field(..., description="Evidence provenance type")
    source_reference: str = Field(..., description="Source identifier, citation, or feed name")
    observed_at: Optional[str] = Field(None, description="Observation ISO timestamp if known; None if unverified")
    published_at: Optional[str] = Field(None, description="Publication ISO timestamp if known; None if unverified")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional verified attributes")

    @field_validator("text")
    @classmethod
    def validate_text_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Evidence text cannot be empty or whitespace only")
        return v.strip()


class CorpusStatistics(BaseModel):
    """Statistical summary of the generated evidence corpus."""
    total_records: int
    source_type_distribution: Dict[str, int]
    task_distribution: Dict[str, int]
    disaster_type_distribution: Dict[str, int]
    modality_distribution: Dict[str, int]
    total_images_referenced: int
    generated_at: str
