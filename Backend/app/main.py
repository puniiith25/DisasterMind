"""
DisasterMind: Multimodal Retrieval-Augmented Generation (RAG) API.

Provides online inference endpoints for:
- Full multimodal RAG inference (DM-07 / DM-08)
- Reciprocal Rank Fusion (RRF) multimodal retrieval (DM-06)
- Dense text retrieval via Qdrant & BGE-M3 (DM-05)
- Direct Qwen2.5-VL vision-language baseline generation (DM-02)
- Dataset validation & corpus statistics (DM-01, DM-03)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.schemas.rag import RAGRequest, RAGResponse
from app.schemas.retrieval import RetrievalResponse
from app.schemas.vlm import VLMInferenceRequest, VLMInferenceResponse
from app.services.multimodal_service import (
    MultimodalRetrievalRequest,
    MultimodalRetrievalResponse,
)
from app.services.qdrant_service import QdrantIndexService
from app.services.rag_service import RAGPipelineService, get_rag_service
from app.services.vlm_service import QwenVLMService

app = FastAPI(
    title="DisasterMind API",
    description="Multimodal Retrieval-Augmented Generation for Disaster Assessment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow CORS for frontends (e.g., local React / Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency Providers (Overridable in tests)
# ---------------------------------------------------------------------------


def get_rag_pipeline() -> RAGPipelineService:
    """Dependency that yields the singleton RAGPipelineService."""
    return get_rag_service()


def get_multimodal_retrieval() -> Any:
    """Dependency that yields the singleton MultimodalRetrievalService."""
    from app.services.multimodal_service import get_multimodal_service
    return get_multimodal_service()


def get_qdrant_index() -> QdrantIndexService:
    """Dependency that yields the singleton QdrantIndexService."""
    from app.services.qdrant_service import get_qdrant_service
    return get_qdrant_service()


def get_vlm() -> QwenVLMService:
    """Dependency that yields the singleton QwenVLMService."""
    from app.services.vlm_service import get_vlm_service
    return get_vlm_service()


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------


class DenseRetrievalQuery(BaseModel):
    """Input payload for dense vector search."""
    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(default=5, ge=1, le=50, description="Max results to return")


class HealthStatus(BaseModel):
    """System health check response."""
    status: str
    version: str
    qdrant_indexed: bool
    qdrant_points: int


# ---------------------------------------------------------------------------
# Routes: Core & Health
# ---------------------------------------------------------------------------


@app.get("/", tags=["General"])
def root() -> Dict[str, Any]:
    """Root endpoint verifying API server status."""
    return {
        "message": "DisasterMind API Server Running",
        "version": "1.0.0",
        "endpoints": {
            "rag": "/api/v1/rag/predict",
            "multimodal_retrieval": "/api/v1/retrieval/multimodal",
            "dense_retrieval": "/api/v1/retrieval/dense",
            "vlm": "/api/v1/vlm/generate",
            "dataset_stats": "/api/v1/dataset/stats",
            "health": "/health",
            "documentation": "/docs",
        },
    }


@app.get("/health", response_model=HealthStatus, tags=["General"])
def health(
    qdrant_svc: QdrantIndexService = Depends(get_qdrant_index),
) -> HealthStatus:
    """Health check endpoint checking vector store index status."""
    try:
        info = qdrant_svc.collection_info()
        exists = bool(info.get("exists", False))
        points = int(info.get("points_count", 0))
    except Exception:
        exists = False
        points = 0

    return HealthStatus(
        status="HEALTHY",
        version="1.0.0",
        qdrant_indexed=exists,
        qdrant_points=points,
    )


# ---------------------------------------------------------------------------
# Routes: Multimodal RAG Pipeline (DM-07 / DM-08)
# ---------------------------------------------------------------------------


@app.post(
    "/api/v1/rag/predict",
    response_model=RAGResponse,
    summary="End-to-End Multimodal RAG Prediction",
    tags=["RAG Pipeline"],
)
def rag_predict(
    request: RAGRequest,
    rag_svc: RAGPipelineService = Depends(get_rag_pipeline),
) -> RAGResponse:
    """
    Execute the full Retrieve-Augment-Generate pipeline:
    1. Retrieve relevant evidence using dense text + optional CLIP image retrieval.
    2. Fuse evidence with Reciprocal Rank Fusion (RRF).
    3. Construct structured prompt with context constraints.
    4. Generate grounded disaster prediction using Qwen2.5-VL.
    """
    try:
        response = rag_svc.run(request)
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline execution failed: {str(exc)}",
        ) from exc


# ---------------------------------------------------------------------------
# Routes: Retrieval Services (DM-05 / DM-06)
# ---------------------------------------------------------------------------


@app.post(
    "/api/v1/retrieval/multimodal",
    response_model=MultimodalRetrievalResponse,
    summary="Multimodal Retrieval (Text + CLIP RRF Fusion)",
    tags=["Retrieval"],
)
def retrieve_multimodal(
    request: MultimodalRetrievalRequest,
    mm_svc: Any = Depends(get_multimodal_retrieval),
) -> MultimodalRetrievalResponse:
    """
    Retrieve ranked disaster evidence using Reciprocal Rank Fusion of:
    - BGE-M3 text dense vectors
    - OpenAI CLIP visual embeddings
    """
    try:
        return mm_svc.retrieve(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multimodal retrieval failed: {str(exc)}",
        ) from exc


@app.post(
    "/api/v1/retrieval/dense",
    response_model=RetrievalResponse,
    summary="Dense Text Retrieval via Qdrant",
    tags=["Retrieval"],
)
def retrieve_dense(
    query_body: DenseRetrievalQuery,
    qdrant_svc: QdrantIndexService = Depends(get_qdrant_index),
) -> RetrievalResponse:
    """Perform dense vector similarity search over DisasterM3 evidence corpus."""
    try:
        return qdrant_svc.retrieve(query=query_body.query, top_k=query_body.top_k)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dense retrieval failed: {str(exc)}",
        ) from exc


# ---------------------------------------------------------------------------
# Routes: Vision-Language Model Baseline (DM-02)
# ---------------------------------------------------------------------------


@app.post(
    "/api/v1/vlm/generate",
    response_model=VLMInferenceResponse,
    summary="Direct VLM Baseline Generation (No RAG)",
    tags=["VLM Baseline"],
)
def vlm_generate(
    request: VLMInferenceRequest,
    vlm_svc: QwenVLMService = Depends(get_vlm),
) -> VLMInferenceResponse:
    """Direct inference using Qwen2.5-VL-3B-Instruct without retrieval augmentation."""
    try:
        return vlm_svc.generate(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"VLM generation failed: {str(exc)}",
        ) from exc


# ---------------------------------------------------------------------------
# Routes: Dataset & Corpus Statistics (DM-01, DM-03)
# ---------------------------------------------------------------------------


@app.get(
    "/api/v1/dataset/stats",
    summary="DisasterM3 Dataset & Evidence Corpus Statistics",
    tags=["Dataset"],
)
def dataset_statistics() -> Dict[str, Any]:
    """Return dataset validation metrics and multimodal corpus statistics."""
    base_dir = Path(__file__).resolve().parent.parent

    validation_file = base_dir / "data" / "reports" / "validation_report.json"
    corpus_file = base_dir / "data" / "reports" / "corpus_statistics.json"

    val_data = {}
    if validation_file.exists():
        try:
            with open(validation_file, "r", encoding="utf-8") as f:
                val_data = json.load(f)
        except Exception:
            val_data = {"error": "Failed to read validation report"}

    corpus_data = {}
    if corpus_file.exists():
        try:
            with open(corpus_file, "r", encoding="utf-8") as f:
                corpus_data = json.load(f)
        except Exception:
            corpus_data = {"error": "Failed to read corpus statistics"}

    return {
        "validation_report": val_data,
        "corpus_statistics": corpus_data,
    }