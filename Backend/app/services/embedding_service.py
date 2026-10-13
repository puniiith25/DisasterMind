"""
BGE-M3 Dense Text Embedding Service for DisasterMind.

Features:
- Configurable BAAI/bge-m3 model loading with device selection (MPS / CUDA / CPU)
- L2-normalized 1024-dimensional dense vectors
- Finite-value and dimension validation (detects NaNs/Infs)
- Safe empty-text normalization
- Persistent disk-based SQLite embedding cache with incremental lookup
- Bounded batch generation tailored for 8 GB RAM constraints
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from app.schemas.embedding import (
    EmbeddingBatchResult,
    EmbeddingMetadata,
    EmbeddingRepresentation,
)

logger = logging.getLogger(__name__)


def select_embedding_device(preferred: str = "auto") -> str:
    pref = preferred.lower().strip()
    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class EmbeddingCache:
    """Persistent SQLite cache for dense vectors keyed by content hash."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get(self, cache_key: str) -> Optional[List[float]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT vector FROM embedding_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def put_batch(self, items: List[Tuple[str, str, List[float]]]) -> None:
        """Batch insert cache items: (cache_key, model_id, vector)."""
        if not items:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embedding_cache (cache_key, model_id, vector)
                VALUES (?, ?, ?)
                """,
                [(k, m, json.dumps(v)) for k, m, v in items],
            )
            conn.commit()


class BGEM3EmbeddingService:
    """Production BGE-M3 text embedding service with persistent caching."""

    def __init__(
        self,
        model_id: str = "BAAI/bge-m3",
        device: str = "auto",
        dimension: int = 1024,
        normalize_embeddings: bool = True,
        cache_dir: Optional[Path | str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = select_embedding_device(device)
        self.dimension = dimension
        self.normalize_embeddings = normalize_embeddings

        if cache_dir:
            self.cache_dir = Path(cache_dir).resolve()
        else:
            self.cache_dir = Path("data/cache/embeddings").resolve()

        self.cache = EmbeddingCache(self.cache_dir / "bge_m3_cache.sqlite3")

        self.model: Any = None
        self.is_loaded: bool = False
        self.load_time_seconds: float = 0.0

    def load_model(self) -> None:
        """Load SentenceTransformer model into memory if not loaded."""
        if self.is_loaded and self.model is not None:
            return

        from sentence_transformers import SentenceTransformer

        start_time = time.perf_counter()
        logger.info(f"Loading {self.model_id} on {self.device}...")

        try:
            self.model = SentenceTransformer(
                self.model_id,
                device=self.device,
            )
        except Exception as e:
            if self.device != "cpu":
                logger.warning(f"Failed to load on {self.device}, falling back to CPU: {e}")
                self.device = "cpu"
                self.model = SentenceTransformer(
                    self.model_id,
                    device="cpu",
                )
            else:
                raise

        self.load_time_seconds = time.perf_counter() - start_time
        self.is_loaded = True
        logger.info(f"Loaded {self.model_id} in {self.load_time_seconds:.2f}s")

    def _make_cache_key(self, text: str) -> str:
        """Create deterministic key combining normalized text and model configuration."""
        raw = f"{self.model_id}__{self.dimension}__{self.normalize_embeddings}__{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _normalize_text(self, text: Any) -> str:
        """Clean input text and handle empty / non-string cases safely."""
        if text is None:
            return "[EMPTY]"
        s = str(text).strip()
        if not s:
            return "[EMPTY]"
        return s

    def _validate_vector(self, vec: List[float]) -> None:
        """Validate vector dimensionality and assert all elements are finite numbers."""
        if len(vec) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(vec)}")
        arr = np.array(vec, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Non-finite values (NaN or Inf) detected in embedding vector")

    def embed_text(self, text: str) -> List[float]:
        """Compute or retrieve dense embedding for a single text."""
        norm_text = self._normalize_text(text)
        cache_key = self._make_cache_key(norm_text)

        cached_vec = self.cache.get(cache_key)
        if cached_vec is not None:
            self._validate_vector(cached_vec)
            return cached_vec

        self.load_model()
        vector_np = self.model.encode(
            norm_text,
            normalize_embeddings=self.normalize_embeddings,
        )
        arr = np.asarray(vector_np, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[0]
        vec = arr.tolist()
        self._validate_vector(vec)
        self.cache.put_batch([(cache_key, self.model_id, vec)])
        return vec

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> Tuple[List[List[float]], EmbeddingBatchResult]:
        """
        Embed a batch of texts incrementally using the cache.
        Bounded batch size limits memory consumption on 8 GB Mac.
        """
        start_time = time.perf_counter()
        norm_texts = [self._normalize_text(t) for t in texts]

        results: List[Optional[List[float]]] = [None] * len(norm_texts)
        missing_indices: List[int] = []
        missing_texts: List[str] = []
        missing_keys: List[str] = []

        cached_count = 0

        # Step 1: Check cache for each text
        for idx, t in enumerate(norm_texts):
            ckey = self._make_cache_key(t)
            cached = self.cache.get(ckey)
            if cached is not None:
                self._validate_vector(cached)
                results[idx] = cached
                cached_count += 1
            else:
                missing_indices.append(idx)
                missing_texts.append(t)
                missing_keys.append(ckey)

        # Step 2: Compute uncached texts in bounded batches
        computed_count = len(missing_texts)
        if computed_count > 0:
            self.load_model()
            cache_entries: List[Tuple[str, str, List[float]]] = []

            for i in range(0, computed_count, batch_size):
                chunk_texts = missing_texts[i : i + batch_size]
                chunk_keys = missing_keys[i : i + batch_size]
                chunk_indices = missing_indices[i : i + batch_size]

                chunk_vectors = self.model.encode(
                    chunk_texts,
                    batch_size=len(chunk_texts),
                    normalize_embeddings=self.normalize_embeddings,
                )

                for orig_idx, ckey, vec_np in zip(chunk_indices, chunk_keys, chunk_vectors):
                    vec = vec_np.tolist()
                    self._validate_vector(vec)
                    results[orig_idx] = vec
                    cache_entries.append((ckey, self.model_id, vec))

            self.cache.put_batch(cache_entries)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        metadata = EmbeddingMetadata(
            model_id=self.model_id,
            dimension=self.dimension,
            representation=EmbeddingRepresentation.DENSE,
            normalized=self.normalize_embeddings,
            device=self.device,
        )

        batch_result = EmbeddingBatchResult(
            count=len(texts),
            cached_count=cached_count,
            computed_count=computed_count,
            latency_ms=latency_ms,
            metadata=metadata,
        )

        final_vectors: List[List[float]] = [v for v in results if v is not None]  # type: ignore[misc]
        return final_vectors, batch_result


_EMBEDDING_SERVICE_INSTANCE: Optional[BGEM3EmbeddingService] = None


def get_embedding_service(
    model_id: str = "BAAI/bge-m3",
    device: str = "auto",
) -> BGEM3EmbeddingService:
    global _EMBEDDING_SERVICE_INSTANCE
    if _EMBEDDING_SERVICE_INSTANCE is None:
        _EMBEDDING_SERVICE_INSTANCE = BGEM3EmbeddingService(model_id=model_id, device=device)
    return _EMBEDDING_SERVICE_INSTANCE
