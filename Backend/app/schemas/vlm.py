"""Pydantic schemas for Qwen2.5-VL Vision-Language inference and evaluation."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class VLMInferenceRequest(BaseModel):
    """Request payload for vision-language inference."""
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(..., min_length=1, description="Disaster question prompt")
    pre_image_path: Optional[str] = Field(None, description="Path to pre-disaster image")
    post_image_path: Optional[str] = Field(None, description="Path to post-disaster image")
    image_path: Optional[str] = Field(None, description="Path to single image")
    image_paths: Optional[List[str]] = Field(default=None, description="Paths to multiple disaster images")
    max_new_tokens: int = Field(default=128, ge=1, le=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Generation temperature")


class VLMInferenceResponse(BaseModel):
    """Structured response from vision-language inference."""
    prediction: str = Field(..., description="Generated text response from the model")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    device: str = Field(..., description="Device used for execution (mps, cuda, cpu)")
    model_id: str = Field(..., description="Model identifier")
    status: str = Field(default="SUCCESS", description="'SUCCESS' or 'ERROR'")
    error_message: Optional[str] = Field(None, description="Error details if inference failed")


class VLMBenchmarkSampleResult(BaseModel):
    """Per-sample benchmark evaluation outcome."""
    sample_id: str
    task: str
    question: str
    ground_truth: str
    prediction: str
    latency_ms: float
    memory_mb: Optional[float] = None
    pre_image: Optional[str] = None
    post_image: Optional[str] = None
    image: Optional[str] = None


class VLMBenchmarkReport(BaseModel):
    """Summary report of baseline VLM performance measurements."""
    timestamp: str
    model_id: str
    device: str
    load_time_seconds: float
    samples_count: int
    avg_latency_ms: float
    peak_memory_mb: Optional[float] = None
    samples: List[VLMBenchmarkSampleResult]
