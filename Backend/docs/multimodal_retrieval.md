# Topic 06: Multimodal Retrieval with Reciprocal Rank Fusion

## 1. Objective
To fuse dense semantic text retrieval with visual image similarity search over DisasterM3 multi-modal evidence using Reciprocal Rank Fusion (RRF).

---

## 2. Architecture & Fusion Pipeline

```
Query (Question + optional Image)
        ├── Text query  → BGE-M3 (1024-d) → Qdrant Cosine Search → text_results (rank 1..K)
        └── Image query → OpenAI CLIP (512-d) → Image Vector Search → image_results (rank 1..K)
                                     │
                                     ▼
                            Reciprocal Rank Fusion (RRF)
                   RRF_score(d) = Σ w_m / (k + rank_m(d))
                                     │
                                     ▼
                        Fused Evidence List (Deduplicated)
```

### Fusion Modes:
- `text_only`: When no query image is supplied.
- `image_only`: When no question text is supplied.
- `multimodal_rrf`: When both query text and imagery are provided.
- Graceful degradation: If image embedding/retrieval fails, automatically falls back to `text_only`.

---

## 3. Schemas & Service

### Data Models (`app/services/multimodal_service.py`)
- `MultimodalRetrievalRequest`: Input query with text and optional image path, custom weights, and top-K limits.
- `FusedEvidenceResult`: Ranked evidence unit with RRF score, text/image individual ranks, and provenance.
- `MultimodalRetrievalResponse`: Full response with latency, fusion mode, and ranked results.

### Service (`MultimodalRetrievalService`)
- Integrates `QdrantIndexService` and `CLIPImageService`.
- Exposes `retrieve(request: MultimodalRetrievalRequest) -> MultimodalRetrievalResponse`.

---

## 4. Execution Commands

### CLI Retrieval Tool
```bash
cd Backend
venv/bin/python scripts/run_multimodal_retrieval.py --question "Earthquake building damage" --fusion-top-k 5
```

### Run Unit Tests
```bash
cd Backend
venv/bin/pytest tests/unit/test_multimodal_service.py -v
```
