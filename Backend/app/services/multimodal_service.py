"""
Multimodal Retrieval Service for DisasterMind.

Combines dense text retrieval (BGE-M3 / Qdrant) with CLIP image retrieval
using Reciprocal Rank Fusion (RRF) to produce a unified ranked evidence list.

Pipeline:
  Query (text + optional image)
       ├── BGE-M3 → Qdrant text search → text_results
       └── CLIP   → Qdrant image search → image_results
                                  │
                                  ▼
                          RRF fusion → unified ranked results
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas (local; also exported via app.schemas)
# ---------------------------------------------------------------------------


class MultimodalRetrievalRequest(BaseModel):
    """Input to the multimodal retrieval pipeline."""
    question: str
    task_type: Optional[str] = None
    image_path: Optional[str] = None
    text_top_k: int = 5
    image_top_k: int = 5
    fusion_top_k: int = 5
    rrf_k: int = 60  # RRF constant (standard: 60)
    text_weight: float = 0.6
    image_weight: float = 0.4


class FusedEvidenceResult(BaseModel):
    """Single fused retrieval result from multimodal RRF."""
    rank: int
    rrf_score: float
    text_rank: Optional[int] = None
    text_score: Optional[float] = None
    image_rank: Optional[int] = None
    image_score: Optional[float] = None
    evidence_id: str
    record_id: str
    task_type: str
    source_type: str
    text: str
    question: str
    answer: str
    disaster_type: Optional[str] = None
    location: Optional[str] = None
    has_images: bool = False
    image_paths: List[str] = Field(default_factory=list)


class MultimodalRetrievalResponse(BaseModel):
    """Complete multimodal retrieval response."""
    question: str
    task_type: Optional[str]
    image_path: Optional[str]
    fusion_mode: str  # "text_only" | "image_only" | "multimodal_rrf"
    text_results_count: int
    image_results_count: int
    fused_results: List[FusedEvidenceResult]
    latency_ms: float


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    text_results: List[Any],
    image_results: List[Any],
    top_k: int = 5,
    rrf_k: int = 60,
    text_weight: float = 0.6,
    image_weight: float = 0.4,
) -> List[Tuple[str, float, Optional[int], Optional[float], Optional[int], Optional[float]]]:
    """
    Merge text and image ranked lists using Reciprocal Rank Fusion.

    RRF score for a document d across ranked lists L:
        RRF(d) = Σ_L  weight_L / (k + rank_L(d))

    Returns list of (evidence_id, rrf_score, text_rank, text_score, image_rank, image_score)
    sorted by descending rrf_score.
    """
    scores: Dict[str, Dict[str, Any]] = {}

    for rank, result in enumerate(text_results, start=1):
        eid = result.evidence_id
        if eid not in scores:
            scores[eid] = {
                "rrf": 0.0,
                "text_rank": None,
                "text_score": None,
                "image_rank": None,
                "image_score": None,
                "result": result,
            }
        scores[eid]["rrf"] += text_weight / (rrf_k + rank)
        scores[eid]["text_rank"] = rank
        scores[eid]["text_score"] = result.score

    for rank, result in enumerate(image_results, start=1):
        eid = result.evidence_id
        if eid not in scores:
            scores[eid] = {
                "rrf": 0.0,
                "text_rank": None,
                "text_score": None,
                "image_rank": None,
                "image_score": None,
                "result": result,
            }
        scores[eid]["rrf"] += image_weight / (rrf_k + rank)
        scores[eid]["image_rank"] = rank
        scores[eid]["image_score"] = result.score

    sorted_items = sorted(scores.items(), key=lambda x: x[1]["rrf"], reverse=True)

    fused = []
    for eid, data in sorted_items[:top_k]:
        fused.append((
            eid,
            data["rrf"],
            data["text_rank"],
            data["text_score"],
            data["image_rank"],
            data["image_score"],
            data["result"],
        ))

    return fused


# ---------------------------------------------------------------------------
# Multimodal Retrieval Service
# ---------------------------------------------------------------------------


class MultimodalRetrievalService:
    """
    Unified multimodal retrieval service combining text + image evidence.

    Supports three retrieval modes:
    - text_only: no image provided → only BGE-M3/Qdrant text retrieval
    - image_only: no question provided → only CLIP image retrieval
    - multimodal_rrf: both provided → RRF fusion of text + image results
    """

    def __init__(
        self,
        qdrant_service: Any,
        clip_service: Any,
        image_collection_name: str = "disasterm3_images",
    ) -> None:
        self.qdrant_service = qdrant_service
        self.clip_service = clip_service
        self.image_collection_name = image_collection_name

    def retrieve(
        self,
        request: MultimodalRetrievalRequest,
    ) -> MultimodalRetrievalResponse:
        """
        Execute multimodal retrieval for a given question + optional image.

        Args:
            request: MultimodalRetrievalRequest with query parameters

        Returns:
            MultimodalRetrievalResponse with fused ranked evidence
        """
        start_time = time.perf_counter()

        has_question = bool(request.question.strip())
        has_image = request.image_path is not None and Path(str(request.image_path)).exists()

        text_results = []
        image_results = []

        # --- Text retrieval (BGE-M3 / Qdrant text collection) ---
        if has_question:
            text_resp = self.qdrant_service.retrieve(
                query=request.question,
                top_k=request.text_top_k,
            )
            text_results = text_resp.results
            logger.debug(f"Text retrieval: {len(text_results)} results")

        # --- Image retrieval (CLIP / Qdrant image collection) ---
        if has_image:
            try:
                image_results = self._retrieve_by_image(
                    image_path=str(request.image_path),
                    top_k=request.image_top_k,
                )
                logger.debug(f"Image retrieval: {len(image_results)} results")
            except Exception as e:
                logger.warning(f"Image retrieval failed (continuing text-only): {e}")

        # --- Determine fusion mode ---
        if text_results and image_results:
            fusion_mode = "multimodal_rrf"
            fused_raw = reciprocal_rank_fusion(
                text_results=text_results,
                image_results=image_results,
                top_k=request.fusion_top_k,
                rrf_k=request.rrf_k,
                text_weight=request.text_weight,
                image_weight=request.image_weight,
            )
        elif text_results:
            fusion_mode = "text_only"
            fused_raw = [
                (
                    r.evidence_id,
                    r.score,
                    i + 1,
                    r.score,
                    None,
                    None,
                    r,
                )
                for i, r in enumerate(text_results[: request.fusion_top_k])
            ]
        else:
            fusion_mode = "image_only"
            fused_raw = [
                (
                    r.evidence_id,
                    r.score,
                    None,
                    None,
                    i + 1,
                    r.score,
                    r,
                )
                for i, r in enumerate(image_results[: request.fusion_top_k])
            ]

        # --- Build structured fused results ---
        fused_results: List[FusedEvidenceResult] = []
        for final_rank, item in enumerate(fused_raw, start=1):
            eid, rrf_score, t_rank, t_score, i_rank, i_score, result = item
            payload = result.payload if hasattr(result, "payload") else {}
            images_list = payload.get("images", [])
            image_paths = [
                img.get("path", img) if isinstance(img, dict) else img
                for img in images_list
            ]

            fused_results.append(
                FusedEvidenceResult(
                    rank=final_rank,
                    rrf_score=rrf_score,
                    text_rank=t_rank,
                    text_score=t_score,
                    image_rank=i_rank,
                    image_score=i_score,
                    evidence_id=result.evidence_id,
                    record_id=result.record_id,
                    task_type=result.task_type,
                    source_type=result.source_type,
                    text=result.text,
                    question=result.question,
                    answer=result.answer,
                    disaster_type=result.disaster_type,
                    location=result.location,
                    has_images=bool(images_list),
                    image_paths=image_paths,
                )
            )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return MultimodalRetrievalResponse(
            question=request.question,
            task_type=request.task_type,
            image_path=request.image_path,
            fusion_mode=fusion_mode,
            text_results_count=len(text_results),
            image_results_count=len(image_results),
            fused_results=fused_results,
            latency_ms=latency_ms,
        )

    def _retrieve_by_image(
        self,
        image_path: str,
        top_k: int = 5,
    ) -> List[Any]:
        """
        Retrieve similar evidence using CLIP image embedding.
        Falls back to text collection if image collection is not available.
        """
        from app.services.qdrant_service import QdrantIndexService
        from app.schemas.retrieval import RetrievalResult

        img_vector = self.clip_service.embed_image(image_path)

        # Try image collection first, fall back to text collection
        client = self.qdrant_service._get_client()

        target_collection = self.image_collection_name
        if not client.collection_exists(collection_name=target_collection):
            logger.warning(
                f"Image collection '{target_collection}' not found; "
                "skipping image retrieval."
            )
            return []

        hits = client.query_points(
            collection_name=target_collection,
            query=img_vector,
            limit=top_k,
        ).points

        results = []
        for rank, hit in enumerate(hits, start=1):
            p = hit.payload or {}
            results.append(
                RetrievalResult(
                    rank=rank,
                    score=float(hit.score),
                    evidence_id=p.get("evidence_id", f"img_{rank}"),
                    record_id=p.get("record_id", ""),
                    task_type=p.get("task_type", ""),
                    source_type=p.get("source_type", "image"),
                    text=p.get("text", ""),
                    question=p.get("question", ""),
                    answer=p.get("answer", ""),
                    disaster_type=p.get("disaster_type"),
                    location=p.get("location"),
                    payload=p,
                )
            )

        return results


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_MULTIMODAL_SERVICE_INSTANCE: Optional[MultimodalRetrievalService] = None


def get_multimodal_service(
    qdrant_path: str = "data/qdrant",
    collection_name: str = "disasterm3_text",
    device: str = "auto",
) -> MultimodalRetrievalService:
    global _MULTIMODAL_SERVICE_INSTANCE
    if _MULTIMODAL_SERVICE_INSTANCE is None:
        from app.services.qdrant_service import QdrantIndexService
        from app.services.clip_service import CLIPImageService

        qdrant_svc = QdrantIndexService(
            qdrant_path=qdrant_path,
            collection_name=collection_name,
        )
        clip_svc = CLIPImageService(device=device)

        _MULTIMODAL_SERVICE_INSTANCE = MultimodalRetrievalService(
            qdrant_service=qdrant_svc,
            clip_service=clip_svc,
        )
    return _MULTIMODAL_SERVICE_INSTANCE
