"""
Unit tests for BGE-M3 Dense Embedding Service and EmbeddingCache.
"""

import math
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from app.services.embedding_service import (
    BGEM3EmbeddingService,
    EmbeddingCache,
    select_embedding_device,
)


class TestEmbeddingService:
    """Test suite for BGEM3EmbeddingService."""

    def test_select_embedding_device(self):
        assert select_embedding_device("cpu") == "cpu"
        assert select_embedding_device("mps") == "mps"
        assert select_embedding_device("cuda") == "cuda"
        assert select_embedding_device("auto") in ("mps", "cuda", "cpu")

    def test_normalize_text(self, tmp_path):
        service = BGEM3EmbeddingService(cache_dir=tmp_path)
        assert service._normalize_text("  hello world  ") == "hello world"
        assert service._normalize_text("") == "[EMPTY]"
        assert service._normalize_text(None) == "[EMPTY]"
        assert service._normalize_text("   \n\t  ") == "[EMPTY]"

    def test_validate_vector_dimension_mismatch(self, tmp_path):
        service = BGEM3EmbeddingService(dimension=1024, cache_dir=tmp_path)
        bad_dim = [0.1] * 512
        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            service._validate_vector(bad_dim)

    def test_validate_vector_non_finite_detected(self, tmp_path):
        service = BGEM3EmbeddingService(dimension=4, cache_dir=tmp_path)
        with pytest.raises(ValueError, match="Non-finite values"):
            service._validate_vector([0.1, float("nan"), 0.3, 0.4])

        with pytest.raises(ValueError, match="Non-finite values"):
            service._validate_vector([0.1, float("inf"), 0.3, 0.4])

    def test_embedding_cache_operations(self, tmp_path):
        cache = EmbeddingCache(tmp_path / "test_cache.sqlite3")
        dummy_vec = [0.1, 0.2, 0.3]

        assert cache.get("key1") is None
        cache.put_batch([("key1", "test_model", dummy_vec)])

        retrieved = cache.get("key1")
        assert retrieved == dummy_vec

    def test_mocked_embed_batch_with_incremental_caching(self, tmp_path):
        service = BGEM3EmbeddingService(dimension=4, cache_dir=tmp_path)
        service.is_loaded = True

        mock_model = MagicMock()
        # Mock encode to return dummy 4-dimensional normalized vectors
        mock_model.encode.side_effect = lambda texts, **kw: np.array([[0.5] * 4 for _ in texts], dtype=np.float32)
        service.model = mock_model

        texts = ["Earthquake in Turkey", "Flood in Libya", "Tornado in Joplin"]

        # Run 1: All 3 are computed
        vectors_1, res_1 = service.embed_batch(texts, batch_size=2)
        assert len(vectors_1) == 3
        assert res_1.computed_count == 3
        assert res_1.cached_count == 0
        assert mock_model.encode.call_count == 2  # 2 chunks because batch_size=2

        # Run 2: Same texts should hit cache with 0 re-computations
        mock_model.encode.reset_mock()
        vectors_2, res_2 = service.embed_batch(texts, batch_size=2)
        assert len(vectors_2) == 3
        assert res_2.computed_count == 0
        assert res_2.cached_count == 3
        assert mock_model.encode.call_count == 0  # Zero calls to model!

        # Run 3: 1 new text and 1 existing
        mock_model.encode.reset_mock()
        mixed_texts = ["Earthquake in Turkey", "Wildfire in California"]
        vectors_3, res_3 = service.embed_batch(mixed_texts, batch_size=2)
        assert len(vectors_3) == 2
        assert res_3.cached_count == 1
        assert res_3.computed_count == 1
        assert mock_model.encode.call_count == 1

    def test_empty_text_handled_safely(self, tmp_path):
        service = BGEM3EmbeddingService(dimension=4, cache_dir=tmp_path)
        service.is_loaded = True

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        service.model = mock_model

        vec = service.embed_text("")
        assert len(vec) == 4
        # Verify the model received '[EMPTY]'
        mock_model.encode.assert_called_once_with("[EMPTY]", normalize_embeddings=True)
