# Topic 04: BGE-M3 Dense Embeddings & Caching

## 1. Objective
To build a high-performance, memory-aware embedding service using `BAAI/bge-m3` (1024-dim dense representation) with persistent SQLite caching.
Key features:
- L2-normalized 1024-dimensional dense vectors for cosine similarity.
- Persistent SQLite cache (`Backend/data/embedding_cache.sqlite`) for incremental lookup and sub-millisecond retrieval of cached representations.
- Automatic hardware selection (Apple MPS, CUDA, CPU fallback).
- Dimension and finite-value validation (detecting NaN / Inf).
- Safe empty-text normalization using a sentinel token `[EMPTY]`.
- Bounded batch generation optimized for 8 GB RAM constraints.

---

## 2. Architecture & Schemas

### Data Model (`app/schemas/embedding.py`)
- `EmbeddingRepresentation`: Encapsulates dense vector (1024-dim), sparse weight vector, and multivec token representations.
- `EmbeddingMetadata`: Tracking model ID, dimension, normalization flag, and hardware device.
- `EmbeddingBatchResult`: Batch output with latency metrics, cache hit rate, and device.
- `EmbeddingBenchmarkReport`: Benchmarking metrics for throughput, peak memory, and cache speed.

### Service Layer (`app/services/embedding_service.py`)
- `EmbeddingCache`: SQLite-backed persistent KV cache indexed by SHA-256 hash of text.
- `BGEM3EmbeddingService`:
  - `embed_batch(texts)`: Embeds batch with incremental caching.
  - `embed_single(text)`: Single query embedding.
  - `select_device()`: Auto hardware detection.

---

## 3. Execution Commands

### Generate & Benchmark Embeddings CLI
```bash
cd Backend
venv/bin/python scripts/generate_embeddings.py --limit 100 --batch-size 16
```

### Run Automated Unit Tests
```bash
cd Backend
venv/bin/pytest tests/unit/test_embedding_service.py -v
```
