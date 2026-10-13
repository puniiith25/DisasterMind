"""Strongly typed schemas for DisasterM3 evidence and QA records."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskType(str, Enum):
    """Known task types in the DisasterM3 benchmark."""
    BUILDING_DAMAGE_COUNTING = "Building Damage Counting"
    ROAD_DAMAGE_COUNTING = "Road Damage Counting"
    DISASTER_SCENE_RECOGNITION = "Disaster Scene Recognition"
    DISASTER_BEARING_BODIES_RECOGNITION = "Disaster Bearing Bodies Recognition"
    DISASTER_TYPE_RECOGNITION = "Disaster Type Recognition"
    RELATIONAL_REASONING = "Relational Reasoning"
    OTHER = "Other"


class DisasterM3Record(BaseModel):
    """A validated QA record from DisasterM3."""
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    id: str = Field(..., description="Unique record identifier (e.g. dm3_00000)")
    task: str = Field(..., description="Disaster task category")
    question: str = Field(..., min_length=1, description="Question text")
    answer: str = Field(..., min_length=1, description="Ground truth answer string")
    image_type: Optional[str] = Field(None, description="Modality type (e.g., Optical, SAR)")
    pre_image: Optional[str] = Field(None, description="Relative or absolute path to pre-disaster image")
    post_image: Optional[str] = Field(None, description="Relative or absolute path to post-disaster image")
    image: Optional[str] = Field(None, description="Relative or absolute path to single image")
    options: Optional[List[Union[str, int, float]]] = Field(None, description="Multiple choice options list if applicable")
    answer_option: Optional[str] = Field(None, description="Letter or key corresponding to ground truth")
    description: Optional[str] = Field(None, description="Additional context or classification description")

    @field_validator("id")
    @classmethod
    def validate_id_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Record ID cannot be empty")
        return v.strip()

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Question cannot be empty or whitespace only")
        return v.strip()

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Answer cannot be empty or whitespace only")
        return v.strip()

    @model_validator(mode="after")
    def validate_at_least_one_image_reference(self) -> DisasterM3Record:
        """Ensure record references at least one image source."""
        if not (self.image or self.pre_image or self.post_image):
            raise ValueError("Record must specify at least one image reference (image, pre_image, or post_image)")
        return self

    def get_image_paths(self) -> List[str]:
        """Return all non-null image paths referenced by this record."""
        paths = []
        if self.pre_image:
            paths.append(self.pre_image)
        if self.post_image:
            paths.append(self.post_image)
        if self.image:
            paths.append(self.image)
        return paths


class IssueSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


class ValidationIssue(BaseModel):
    """Structured report of a validation anomaly or failure."""
    record_id: Optional[str] = None
    record_index: int
    field: Optional[str] = None
    issue_type: str = Field(..., description="Machine-readable issue category")
    message: str = Field(..., description="Human-readable explanation")
    severity: IssueSeverity = IssueSeverity.ERROR


class DatasetSplitSummary(BaseModel):
    """Summary of train/dev/test split generation."""
    total_samples: int
    train_count: int
    dev_count: int
    test_count: int
    train_tasks: Dict[str, int]
    dev_tasks: Dict[str, int]
    test_tasks: Dict[str, int]
    seed: int


class ValidationReport(BaseModel):
    """Comprehensive dataset validation report."""
    timestamp: str
    dataset_path: str
    total_records: int
    valid_records: int
    invalid_records: int
    duplicate_ids_count: int
    missing_images_count: int
    task_distribution: Dict[str, int]
    image_type_distribution: Dict[str, int]
    issues: List[ValidationIssue]
    summary_status: str = Field(..., description="'PASS' if valid_records == total_records and no errors, else 'FAIL'")
