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
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.schemas.investigation import (
    ImageMetadataInfo,
    InvestigationGroundedAnswer,
    LocationInvestigationRequest,
    RouteInvestigationRequest,
    RouteInvestigationResponse,
)
from app.schemas.live_feed import LiveFeedIngestRequest
from app.schemas.rag import RAGRequest, RAGResponse
from app.schemas.retrieval import RetrievalResponse
from app.schemas.vlm import VLMInferenceRequest, VLMInferenceResponse
from app.services.multimodal_service import (
    MultimodalRetrievalRequest,
    MultimodalRetrievalResponse,
)
import logging
import threading
from contextlib import asynccontextmanager

from app.services.qdrant_service import QdrantIndexService
from app.services.rag_service import RAGPipelineService, get_rag_service
from app.services.vlm_service import QwenVLMService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up vision model in background thread so user never hits cold-start 20s delay
    def _warmup():
        try:
            from app.services.vlm_service import get_vlm_service
            logger.info("Warming up VLM model in background thread...")
            get_vlm_service().load_model()
            logger.info("VLM model successfully warmed up and ready!")
        except Exception as err:
            logger.warning("VLM background warmup deferred: %s", err)

    threading.Thread(target=_warmup, daemon=True).start()
    yield


app = FastAPI(
    title="DisasterMind API",
    description="Multimodal Retrieval-Augmented Generation for Disaster Assessment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Allow CORS for frontends (e.g., local React / Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for uploaded disaster images
upload_dir = Path(__file__).resolve().parent.parent / "data" / "uploads"
upload_dir.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(upload_dir)), name="uploads")


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


def get_grounding() -> Any:
    """Dependency that yields the singleton GroundingService."""
    from app.services.grounding_service import get_grounding_service
    return get_grounding_service()


def get_live_feed() -> Any:
    """Dependency that yields the singleton LiveFeedService."""
    from app.services.live_feed_service import get_live_feed_service
    return get_live_feed_service()


def get_verification() -> Any:
    """Dependency that yields the singleton EvidenceVerificationService."""
    from app.services.verification_service import get_verification_service
    return get_verification_service()


def get_intelligence() -> Any:
    """Dependency that yields the singleton IncidentIntelligenceService."""
    from app.services.intelligence_service import get_intelligence_service
    return get_intelligence_service()




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


@app.get("/api/v1/vlm/status", tags=["General"])
def vlm_status(
    vlm_svc: QwenVLMService = Depends(get_vlm),
) -> Dict[str, Any]:
    """Return Vision-Language model readiness and runtime accelerator device."""
    ready = getattr(vlm_svc, "is_ready", lambda: getattr(vlm_svc, "is_loaded", False))()
    return {
        "ready": ready,
        "is_loading": getattr(vlm_svc, "_is_loading", False),
        "device": getattr(vlm_svc, "device", "auto"),
        "model_id": getattr(vlm_svc, "model_id", "Qwen/Qwen2.5-VL-3B-Instruct"),
        "load_time_seconds": round(getattr(vlm_svc, "load_time_seconds", 0.0), 2),
    }


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


# ---------------------------------------------------------------------------
# Routes: Grounding, Live Feed & Verification (DM-08, DM-09, DM-10)
# ---------------------------------------------------------------------------


class GroundingEvaluationRequest(BaseModel):
    prediction: str
    evidence_snippets: list[Any] = Field(default_factory=list)


@app.post(
    "/api/v1/grounding/evaluate",
    summary="Evaluate Grounding and Extract Citations",
    tags=["Grounding & Verification"],
)
def evaluate_grounding(
    request: GroundingEvaluationRequest,
    grounding_svc: Any = Depends(get_grounding),
) -> Any:
    """Evaluate prediction against retrieved evidence to compute citation links and hallucination risk."""
    try:
        from app.schemas.rag import EvidenceSnippet
        snippets = [
            EvidenceSnippet(**s) if isinstance(s, dict) else s
            for s in request.evidence_snippets
        ]
        return grounding_svc.evaluate(request.prediction, snippets)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grounding evaluation failed: {str(exc)}",
        ) from exc


@app.post(
    "/api/v1/live/ingest",
    summary="Ingest Live Disaster Evidence",
    tags=["Live Feeds"],
)
def ingest_live_evidence(
    request: LiveFeedIngestRequest,
    live_svc: Any = Depends(get_live_feed),
    qdrant_svc: QdrantIndexService = Depends(get_qdrant_index),
) -> Any:
    """Fetch external live disaster incident bulletins (e.g. USGS) and normalize into evidence corpus."""
    try:
        return live_svc.ingest(request, qdrant_service=qdrant_svc)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Live feed ingestion failed: {str(exc)}",
        ) from exc



class VerificationCheckRequest(BaseModel):
    prediction: str
    evidence_snippets: list[Any] = Field(default_factory=list)


@app.post(
    "/api/v1/verification/verify",
    summary="Verify Prediction and Detect Contradictions",
    tags=["Grounding & Verification"],
)
def verify_prediction(
    request: VerificationCheckRequest,
    verification_svc: Any = Depends(get_verification),
) -> Any:
    """Check prediction claims against evidence for factual support, numerical clashes, and contradictions."""
    try:
        from app.schemas.rag import EvidenceSnippet
        snippets = [
            EvidenceSnippet(**s) if isinstance(s, dict) else s
            for s in request.evidence_snippets
        ]
        return verification_svc.verify(request.prediction, snippets)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evidence verification failed: {str(exc)}",
        ) from exc


# ---------------------------------------------------------------------------
# Routes: Real-time Image & Live Internet Intelligence
# ---------------------------------------------------------------------------


class IntelligenceJSONRequest(BaseModel):
    question: str
    image_path: Optional[str] = None
    search_live: bool = True
    engine_mode: str = "fast"


@app.post(
    "/api/v1/intelligence/analyze",
    summary="Analyze Incident with Live Internet Evidence (JSON)",
    tags=["Intelligence"],
)
def analyze_intelligence_json(
    request: IntelligenceJSONRequest,
    intelligence_svc: Any = Depends(get_intelligence),
) -> Any:
    """Analyze query and optional local image path against live internet disaster bulletins."""
    try:
        return intelligence_svc.analyze_incident(
            question=request.question,
            image_path=request.image_path,
            search_live=request.search_live,
            engine_mode=request.engine_mode,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Intelligence analysis failed: {str(exc)}",
        ) from exc


@app.post(
    "/api/v1/intelligence/analyze-upload",
    summary="Upload Image(s) & Analyze Incident with Live Internet Search",
    tags=["Intelligence"],
)
def analyze_intelligence_upload(
    question: str = Form(...),
    search_live: bool = Form(True),
    engine_mode: str = Form("deep_vlm"),
    conversation_history: Optional[str] = Form(None),
    existing_image_paths: Optional[str] = Form(None),
    images: List[UploadFile] = File(default=[]),
    image: Optional[UploadFile] = File(None),
    intelligence_svc: Any = Depends(get_intelligence),
) -> Any:
    """
    Accepts user uploaded image(s) + natural language question.
    Processes all uploaded scenes, performs real-time visual reasoning and live internet cross-correlation.
    Preserves conversation context across follow-up queries.
    """
    saved_paths: List[str] = []
    try:
        import os
        import time
        upload_dir = Path(__file__).resolve().parent.parent / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        now_ts = int(time.time())

        # Reuse existing paths from previous session turns if provided
        if existing_image_paths:
            try:
                parsed_paths = json.loads(existing_image_paths)
                if isinstance(parsed_paths, list):
                    for ep in parsed_paths:
                        if ep and os.path.exists(str(ep)) and str(ep) not in saved_paths:
                            saved_paths.append(str(ep))
            except Exception:
                for ep in existing_image_paths.split(","):
                    ep = ep.strip()
                    if ep and os.path.exists(ep) and ep not in saved_paths:
                        saved_paths.append(ep)

        # Collect all new uploads (preventing duplicate multi/single file submissions)
        all_uploads: List[UploadFile] = []
        if images:
            all_uploads.extend(images)
        elif image and image.filename:
            all_uploads.append(image)

        seen_names = set()
        for i, up in enumerate(all_uploads):
            if up and up.filename:
                base_name = os.path.basename(up.filename)
                if base_name in seen_names:
                    continue
                seen_names.add(base_name)
                safe_name = f"upload_{now_ts}_{i}_{base_name}"
                p = upload_dir / safe_name
                with open(p, "wb") as f:
                    f.write(up.file.read())
                saved_paths.append(str(p))

        return intelligence_svc.analyze_incident(
            question=question,
            image_path=saved_paths[0] if saved_paths else None,
            image_paths=saved_paths if saved_paths else None,
            search_live=search_live,
            engine_mode=engine_mode,
            conversation_history=conversation_history,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Uploaded image intelligence analysis failed: {str(exc)}",
        ) from exc


# ---------------------------------------------------------------------------
# Routes: Multi-Turn Conversational Vision Chat (ChatGPT-style)
# ---------------------------------------------------------------------------


def get_chat():
    from app.services.chat_service import get_chat_service
    return get_chat_service()


@app.post(
    "/api/v1/chat/message",
    summary="Multi-turn Conversational Disaster Vision Chat",
    tags=["Chat"],
)
def chat_message(
    message: str = Form(...),
    history: str = Form("[]"),
    existing_image_paths: str = Form("[]"),
    engine_mode: str = Form("deep_vlm"),
    images: List[UploadFile] = File(default=[]),
    image: Optional[UploadFile] = File(None),
    chat_svc: Any = Depends(get_chat),
) -> Any:
    """
    Handles conversational multi-turn chat on uploaded disaster imagery.
    Turn 1 receives uploaded image + initial question.
    Subsequent turns answer follow-up questions referencing the previous image and conversation.
    """
    try:
        import time
        upload_dir = Path(__file__).resolve().parent.parent / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        now_ts = int(time.time())

        # Collect existing image paths
        try:
            active_paths: List[str] = json.loads(existing_image_paths) if existing_image_paths else []
        except Exception:
            active_paths = []

        # Save any newly uploaded images
        all_uploads: List[UploadFile] = []
        if images:
            all_uploads.extend(images)
        if image and image.filename:
            all_uploads.append(image)

        for i, up in enumerate(all_uploads):
            if up and up.filename:
                safe_name = f"chat_{now_ts}_{i}_{up.filename}"
                p = upload_dir / safe_name
                with open(p, "wb") as f:
                    f.write(up.file.read())
                active_paths.append(str(p))

        # Parse history
        try:
            parsed_history = json.loads(history) if history else []
        except Exception:
            parsed_history = []

        return chat_svc.chat(
            message=message,
            history=parsed_history,
            image_paths=active_paths,
            engine_mode=engine_mode,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat reasoning failed: {str(exc)}",
        ) from exc


# ---------------------------------------------------------------------------
# Dependency Providers for Current Disaster & Route Intelligence
# ---------------------------------------------------------------------------


def get_investigation() -> Any:
    from app.services.disaster_investigation_service import get_investigation_service
    return get_investigation_service()


def get_route_risk() -> Any:
    from app.services.route_risk_service import get_route_risk_service
    return get_route_risk_service()


def get_metadata() -> Any:
    from app.services.metadata_service import get_metadata_service
    return get_metadata_service()


def get_external_disaster() -> Any:
    from app.services.external_disaster_service import get_external_disaster_service
    return get_external_disaster_service()


# ---------------------------------------------------------------------------
# Routes: Current Disaster Intelligence & Route Risk (Stages 2 & 3)
# ---------------------------------------------------------------------------


@app.post(
    "/api/v1/investigate/image",
    response_model=InvestigationGroundedAnswer,
    summary="Investigate Disaster Image with EXIF GPS & Grounded Answer",
    tags=["Disaster Investigation"],
)
def investigate_image(
    image: UploadFile = File(...),
    question: str = Form("What disaster is occurring here and what damage is visible?"),
    confirmed_location: Optional[str] = Form(None),
    confirmed_lat: Optional[float] = Form(None),
    confirmed_lon: Optional[float] = Form(None),
    investigation_svc: Any = Depends(get_investigation),
) -> InvestigationGroundedAnswer:
    """
    Accepts user uploaded disaster image, validates MIME/size, extracts EXIF GPS,
    correlates with live authoritative disaster feeds and clean historical evidence,
    and returns a standardized 10-point evidence-grounded answer.
    """
    try:
        content_bytes = investigation_svc.validate_image_upload(image)
        return investigation_svc.investigate_image(
            image_bytes=content_bytes,
            filename=image.filename or "upload.jpg",
            question=question,
            confirmed_location=confirmed_location,
            confirmed_lat=confirmed_lat,
            confirmed_lon=confirmed_lon,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image investigation failed: {str(exc)}",
        ) from exc


@app.post(
    "/api/v1/investigate/location",
    response_model=InvestigationGroundedAnswer,
    summary="Investigate Disaster Hazards by Location",
    tags=["Disaster Investigation"],
)
def investigate_location(
    request: LocationInvestigationRequest,
    investigation_svc: Any = Depends(get_investigation),
) -> InvestigationGroundedAnswer:
    """
    Investigates current and historical disaster threats for a user-confirmed location
    across USGS, GDACS, Open-Meteo, and DisasterM3 evidence.
    """
    try:
        return investigation_svc.investigate_location(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Location investigation failed: {str(exc)}",
        ) from exc


@app.post(
    "/api/v1/investigate/route",
    response_model=RouteInvestigationResponse,
    summary="Investigate Transit Route Disaster & Road Hazards",
    tags=["Route Risk Investigation"],
)
def investigate_route(
    request: RouteInvestigationRequest,
    route_svc: Any = Depends(get_route_risk),
) -> RouteInvestigationResponse:
    """
    Computes highway corridor geometry between origin and destination, checks intersecting
    seismic, storm, and multi-hazard bulletins, and generates an honest safety advisory.
    """
    try:
        return route_svc.investigate_route(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Route risk investigation failed: {str(exc)}",
        ) from exc


@app.post(
    "/api/v1/metadata/image",
    response_model=ImageMetadataInfo,
    summary="Extract EXIF Geolocation Metadata from Image",
    tags=["Disaster Investigation"],
)
def extract_image_metadata(
    image: UploadFile = File(...),
    metadata_svc: Any = Depends(get_metadata),
    investigation_svc: Any = Depends(get_investigation),
) -> ImageMetadataInfo:
    """
    Fast extraction of EXIF GPS coordinates, camera model, and capture timestamp
    without executing heavy neural models, enabling interactive UI location confirmation.
    """
    try:
        content_bytes = investigation_svc.validate_image_upload(image)
        return metadata_svc.extract_metadata(content_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metadata extraction failed: {str(exc)}",
        ) from exc


@app.get(
    "/api/v1/geocode/search",
    summary="Search Location Coordinates via OpenStreetMap Nominatim",
    tags=["Disaster Investigation"],
)
def geocode_search(
    query: str,
    external_svc: Any = Depends(get_external_disaster),
) -> Dict[str, Any]:
    """Search coordinates and standardized place names for location confirmation."""
    res = external_svc.geocode(query)
    if not res:
        return {"found": False, "query": query, "result": None}
    return {"found": True, "query": query, "result": res}