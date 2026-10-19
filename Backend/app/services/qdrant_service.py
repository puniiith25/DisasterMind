"""
Qdrant Indexing Service for DisasterMind.

Features:
- Local disk-persisted Qdrant collection (no server required)
- BGE-M3 dense 1024-dim embeddings via BGEM3EmbeddingService
- Idempotent build: recreates collection on each full rebuild
- Incremental upsert with batched writes
- Rich IndexBuildReport for traceability
- Text retriever returning ranked RetrievalResult list
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from app.schemas.retrieval import (
    CollectionStatus,
    IndexBuildReport,
    IndexedPoint,
    RetrievalResponse,
    RetrievalResult,
)
from app.services.embedding_service import BGEM3EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_COLLECTION = "disasterm3_historical"
DEFAULT_VECTOR_SIZE = 1024
DEFAULT_DISTANCE = Distance.COSINE
DEFAULT_BATCH_SIZE = 16


class QdrantIndexService:
    """
    Builds and queries a local Qdrant collection from a standardized evidence corpus.

    The collection is stored as a directory on disk (no Docker required).
    """

    def __init__(
        self,
        qdrant_path: str | Path = "data/qdrant",
        collection_name: str = DEFAULT_COLLECTION,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        embedding_service: Optional[BGEM3EmbeddingService] = None,
    ) -> None:
        self.qdrant_path = Path(qdrant_path).resolve()
        self.collection_name = collection_name
        self.vector_size = vector_size

        self._embedding_service = embedding_service or get_embedding_service()
        self._client: Optional[QdrantClient] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self.qdrant_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self.qdrant_path))
            logger.info(f"Qdrant client opened at {self.qdrant_path}")
        return self._client

    def _ensure_collection(self, recreate: bool = True) -> CollectionStatus:
        client = self._get_client()
        exists = client.collection_exists(collection_name=self.collection_name)

        if exists and recreate:
            client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted existing collection '{self.collection_name}'")
            exists = False

        if not exists:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=DEFAULT_DISTANCE,
                ),
            )
            logger.info(f"Created collection '{self.collection_name}' (dim={self.vector_size}, {DEFAULT_DISTANCE})")
            return CollectionStatus.CREATED

        return CollectionStatus.EXISTS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(
        self,
        corpus_path: str | Path,
        batch_size: int = DEFAULT_BATCH_SIZE,
        recreate: bool = True,
    ) -> IndexBuildReport:
        """
        Build the Qdrant index from the standardized evidence corpus JSON.

        Args:
            corpus_path: Path to standardized_corpus.json
            batch_size:  BGE-M3 embedding batch size
            recreate:    If True, drops and rebuilds the collection

        Returns:
            IndexBuildReport with detailed metrics
        """
        corpus_path = Path(corpus_path).resolve()
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

        with open(corpus_path, "r", encoding="utf-8") as f:
            records: List[Dict[str, Any]] = json.load(f)

        logger.info(f"Loaded {len(records)} records from {corpus_path}")

        status = self._ensure_collection(recreate=recreate)
        client = self._get_client()

        texts = [r.get("text", "") for r in records]
        start_time = time.perf_counter()

        vectors, batch_result = self._embedding_service.embed_batch(
            texts, batch_size=batch_size
        )

        points: List[PointStruct] = []
        indexed_refs: List[IndexedPoint] = []
        skipped = 0

        for idx, (record, vector) in enumerate(zip(records, vectors)):
            eid = record.get("evidence_id", f"evi_{idx:06d}")
            rid = record.get("record_id", f"rec_{idx:06d}")
            payload = {
                "evidence_id": eid,
                "record_id": rid,
                "text": record.get("text", ""),
                "task_type": record.get("task_type", ""),
                "question": record.get("question", ""),
                "answer": record.get("answer", ""),
                "disaster_type": record.get("disaster_type", ""),
                "location": record.get("location", ""),
                "source_type": record.get("source_type", ""),
                "images": record.get("images", []),
            }
            points.append(PointStruct(id=idx, vector=vector, payload=payload))
            indexed_refs.append(
                IndexedPoint(
                    point_id=idx,
                    evidence_id=eid,
                    record_id=rid,
                    task_type=record.get("task_type", ""),
                    source_type=record.get("source_type", ""),
                )
            )

        # Upsert in batches
        UPSERT_BATCH = 256
        for i in range(0, len(points), UPSERT_BATCH):
            client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + UPSERT_BATCH],
            )
            logger.debug(f"Upserted points {i}–{min(i+UPSERT_BATCH, len(points))-1}")

        total_time = time.perf_counter() - start_time
        throughput = len(records) / total_time if total_time > 0 else 0.0

        report = IndexBuildReport(
            collection_name=self.collection_name,
            vector_size=self.vector_size,
            distance_metric=str(DEFAULT_DISTANCE),
            total_records=len(records),
            indexed_count=len(points),
            skipped_count=skipped,
            status=status,
            qdrant_path=str(self.qdrant_path),
            model_id=self._embedding_service.model_id,
            latency_seconds=total_time,
            throughput_items_per_sec=throughput,
            cache_hit_rate=batch_result.cached_count / len(records) if records else 0.0,
            indexed_points=indexed_refs,
        )

        logger.info(
            f"Indexed {len(points)} points in {total_time:.2f}s "
            f"({throughput:.1f} items/s, cache={batch_result.cached_count}/{len(records)})"
        )
        return report

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResponse:
        """
        Retrieve top-k similar evidence records for a query text.

        Args:
            query: Natural-language question or query
            top_k: Number of results to return

        Returns:
            RetrievalResponse with ranked results and latency
        """
        start_time = time.perf_counter()
        client = self._get_client()

        query_vector = self._embedding_service.embed_text(query)

        coll = self.collection_name
        if not client.collection_exists(collection_name=coll):
            if client.collection_exists(collection_name="disasterm3_historical"):
                coll = "disasterm3_historical"
            elif client.collection_exists(collection_name="disasterm3_text"):
                coll = "disasterm3_text"

        hits = client.query_points(
            collection_name=coll,
            query=query_vector,
            limit=top_k,
        ).points

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        results: List[RetrievalResult] = []
        for rank, hit in enumerate(hits, start=1):
            p = hit.payload or {}
            results.append(
                RetrievalResult(
                    rank=rank,
                    score=float(hit.score),
                    evidence_id=str(p.get("evidence_id") or p.get("id") or hit.id),
                    record_id=str(p.get("record_id") or p.get("id") or hit.id),
                    task_type=p.get("task_type") or p.get("task") or "",
                    source_type=p.get("source_type") or p.get("source") or "historical",
                    text=p.get("text") or f"{p.get('question', '')} -> {p.get('answer', '')}",
                    question=p.get("question", ""),
                    answer=p.get("answer", ""),
                    disaster_type=p.get("disaster_type"),
                    location=p.get("location"),
                    payload=p,
                )
            )

        return RetrievalResponse(
            query=query,
            top_k=top_k,
            collection_name=self.collection_name,
            results=results,
            latency_ms=latency_ms,
        )

    def collection_info(self) -> Dict[str, Any]:
        """Return basic metadata about the current collection."""
        client = self._get_client()
        if not client.collection_exists(collection_name=self.collection_name):
            return {"exists": False, "collection_name": self.collection_name}
        info = client.get_collection(collection_name=self.collection_name)
        return {
            "exists": True,
            "collection_name": self.collection_name,
            "vectors_count": info.points_count,
            "points_count": info.points_count,
            "status": str(info.status),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_QDRANT_SERVICE_INSTANCE: Optional[QdrantIndexService] = None


def get_qdrant_service(
    qdrant_path: str | Path = "data/qdrant",
    collection_name: str = DEFAULT_COLLECTION,
) -> QdrantIndexService:
    global _QDRANT_SERVICE_INSTANCE
    if _QDRANT_SERVICE_INSTANCE is None:
        _QDRANT_SERVICE_INSTANCE = QdrantIndexService(
            qdrant_path=qdrant_path,
            collection_name=collection_name,
        )
    return _QDRANT_SERVICE_INSTANCE
