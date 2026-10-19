"""
RAG Pipeline Service for DisasterMind (DM-07).

Full Retrieve-Augment-Generate pipeline:
  1. Multimodal retrieval (BGE-M3 text + CLIP image, RRF fusion)
  2. Evidence context assembly with token-budget enforcement
  3. Qwen2.5-VL generation conditioned on retrieved evidence
  4. Structured RAGResponse with provenance
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional

from app.schemas.rag import (
    EvidenceSnippet,
    RAGContext,
    RAGRequest,
    RAGResponse,
)
from app.schemas.vlm import VLMInferenceRequest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are DisasterMind, an expert disaster assessment AI. "
    "You analyze satellite imagery and answer questions about disaster impacts. "
    "Use the provided historical evidence to support your answer. "
    "Be concise and factual."
)

EVIDENCE_HEADER = "=== HISTORICAL EVIDENCE ===\n"
EVIDENCE_FOOTER = "\n=== END OF EVIDENCE ===\n"


def build_rag_context(
    question: str,
    task_type: Optional[str],
    fused_results: List[Any],
    max_context_chars: int = 3000,
    retrieval_mode: str = "text_only",
) -> RAGContext:
    """
    Assemble a token-budget-aware evidence context block for the VLM.

    Iterates over retrieved results and appends snippets until the
    character budget is reached.

    Args:
        question:          The original user question.
        task_type:         Optional disaster task category.
        fused_results:     Ranked FusedEvidenceResult items.
        max_context_chars: Maximum total characters for evidence block.
        retrieval_mode:    Fusion mode used for this context.

    Returns:
        RAGContext with formatted evidence text and provenance list.
    """
    task_line = f"Task: {task_type}\n" if task_type else ""
    evidence_lines: List[str] = []
    snippets: List[EvidenceSnippet] = []
    used_chars = 0

    for result in fused_results:
        snippet_text = (
            f"[Evidence {result.rank}] "
            f"Task: {result.task_type} | "
            f"Disaster: {result.disaster_type or 'Unknown'} | "
            f"Location: {result.location or 'Unknown'}\n"
            f"Q: {result.question}\n"
            f"A: {result.answer}\n"
        )

        if used_chars + len(snippet_text) > max_context_chars:
            # Truncate last snippet to fit budget
            remaining = max_context_chars - used_chars
            if remaining > 80:
                evidence_lines.append(snippet_text[:remaining] + "...\n")
                snippets.append(
                    EvidenceSnippet(
                        rank=result.rank,
                        evidence_id=result.evidence_id,
                        task_type=result.task_type,
                        source_type=result.source_type,
                        question=result.question,
                        answer=result.answer,
                        rrf_score=result.rrf_score,
                        disaster_type=result.disaster_type,
                        location=result.location,
                    )
                )
            break

        evidence_lines.append(snippet_text)
        snippets.append(
            EvidenceSnippet(
                rank=result.rank,
                evidence_id=result.evidence_id,
                task_type=result.task_type,
                source_type=result.source_type,
                question=result.question,
                answer=result.answer,
                rrf_score=result.rrf_score,
                disaster_type=result.disaster_type,
                location=result.location,
            )
        )
        used_chars += len(snippet_text)

    evidence_block = EVIDENCE_HEADER + "".join(evidence_lines) + EVIDENCE_FOOTER
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"{task_line}"
        f"{evidence_block}\n"
        f"Question: {question.strip()}\n\n"
        f"Answer:"
    )

    return RAGContext(
        system_prompt=SYSTEM_PROMPT,
        evidence_block=evidence_block,
        full_prompt=full_prompt,
        evidence_count=len(snippets),
        evidence_snippets=snippets,
        context_chars=len(evidence_block),
        retrieval_mode=retrieval_mode,
    )


# ---------------------------------------------------------------------------
# RAG Pipeline Service
# ---------------------------------------------------------------------------


class RAGPipelineService:
    """
    End-to-end Retrieve-Augment-Generate pipeline for DisasterMind.

    Chains:
      MultimodalRetrievalService → context builder → QwenVLMService

    Designed for:
      - Offline research evaluation on DisasterM3 benchmark splits
      - Online inference via the FastAPI endpoint (DM-08)
    """

    def __init__(
        self,
        retrieval_service: Any,  # MultimodalRetrievalService
        vlm_service: Any,        # QwenVLMService
    ) -> None:
        self.retrieval_service = retrieval_service
        self.vlm_service = vlm_service

    def run(self, request: RAGRequest) -> RAGResponse:
        """
        Execute the full RAG pipeline for a single query.

        Steps:
          1. Multimodal retrieval (text + optional image)
          2. Evidence context assembly (with char budget)
          3. VLM generation conditioned on evidence
          4. Structured RAGResponse with full provenance

        Args:
            request: RAGRequest with question, images, and configuration.

        Returns:
            RAGResponse with prediction, evidence snippets, and latency.
        """
        total_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Step 1: Retrieval
        # ------------------------------------------------------------------
        from app.services.multimodal_service import MultimodalRetrievalRequest

        # Use post_image_path as the primary retrieval image if provided
        retrieval_image = request.image_path or request.post_image_path

        mm_request = MultimodalRetrievalRequest(
            question=request.question,
            task_type=request.task_type,
            image_path=retrieval_image,
            text_top_k=request.text_top_k,
            image_top_k=request.image_top_k,
            fusion_top_k=request.fusion_top_k,
        )

        retrieval_start = time.perf_counter()
        mm_response = self.retrieval_service.retrieve(mm_request)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

        fused_results = mm_response.fused_results

        if not fused_results:
            total_latency_ms = (time.perf_counter() - total_start) * 1000.0
            return RAGResponse(
                question=request.question,
                task_type=request.task_type,
                prediction="",
                status="NO_EVIDENCE",
                error_message="No evidence retrieved for this query.",
                retrieval_mode=mm_response.fusion_mode,
                evidence_count=0,
                retrieval_latency_ms=retrieval_latency_ms,
                total_latency_ms=total_latency_ms,
            )

        # ------------------------------------------------------------------
        # Step 2: Context assembly
        # ------------------------------------------------------------------
        rag_context = build_rag_context(
            question=request.question,
            task_type=request.task_type,
            fused_results=fused_results,
            max_context_chars=request.max_context_chars,
            retrieval_mode=mm_response.fusion_mode,
        )

        # ------------------------------------------------------------------
        # Step 3: VLM generation
        # ------------------------------------------------------------------
        vlm_request = VLMInferenceRequest(
            question=rag_context.full_prompt,
            pre_image_path=request.pre_image_path,
            post_image_path=request.post_image_path,
            image_path=request.image_path,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )

        vlm_start = time.perf_counter()
        vlm_response = self.vlm_service.generate(vlm_request)
        vlm_latency_ms = (time.perf_counter() - vlm_start) * 1000.0

        total_latency_ms = (time.perf_counter() - total_start) * 1000.0

        return RAGResponse(
            question=request.question,
            task_type=request.task_type,
            prediction=vlm_response.prediction,
            status=vlm_response.status,
            error_message=vlm_response.error_message,
            retrieval_mode=rag_context.retrieval_mode,
            evidence_count=rag_context.evidence_count,
            evidence_snippets=rag_context.evidence_snippets
            if request.include_provenance
            else [],
            context_chars=rag_context.context_chars,
            retrieval_latency_ms=retrieval_latency_ms,
            vlm_latency_ms=vlm_latency_ms,
            total_latency_ms=total_latency_ms,
            vlm_model_id=vlm_response.model_id,
            device=vlm_response.device,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_RAG_SERVICE_INSTANCE: Optional[RAGPipelineService] = None


def get_rag_service(
    qdrant_path: str = "data/qdrant",
    collection_name: str = "disasterm3_text",
    vlm_device: str = "auto",
    embedding_device: str = "auto",
    clip_device: str = "auto",
) -> RAGPipelineService:
    global _RAG_SERVICE_INSTANCE
    if _RAG_SERVICE_INSTANCE is None:
        from app.services.multimodal_service import get_multimodal_service
        from app.services.vlm_service import get_vlm_service

        mm_svc = get_multimodal_service(
            qdrant_path=qdrant_path,
            collection_name=collection_name,
            device=clip_device,
        )
        vlm_svc = get_vlm_service(device=vlm_device)

        _RAG_SERVICE_INSTANCE = RAGPipelineService(
            retrieval_service=mm_svc,
            vlm_service=vlm_svc,
        )
    return _RAG_SERVICE_INSTANCE
