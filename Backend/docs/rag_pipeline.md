# Topic 07: DisasterMind Multimodal RAG Pipeline

## 1. Objective
To orchestrate the full Retrieve-Augment-Generate cycle for disaster visual question answering, combining:
- Multimodal retrieval (`MultimodalRetrievalService`)
- Evidence context synthesis with character budget enforcement
- Vision-language generation with `Qwen2.5-VL-3B-Instruct`
- Comprehensive provenance tracking and timing metrics

---

## 2. Pipeline Stages

1. **Retrieve**: Queries Qdrant dense index and CLIP visual corpus; fuses results via RRF.
2. **Augment**: Formats retrieved disaster evidence into a structured system context block adhering to a token/character budget (`max_context_chars`).
3. **Generate**: Constructs a multimodal chat prompt conditioned on verified evidence snippets and invokes `QwenVLMService`.
4. **Provenance**: Returns prediction alongside full audit trails (evidence IDs, ranks, RRF scores, latencies).

---

## 3. Schemas & Service

### Data Models (`app/schemas/rag.py`)
- `RAGRequest`: Full query input with question, optional pre/post disaster images, retrieval top-k parameters, and generation settings.
- `EvidenceSnippet`: Formatted evidence unit injected into prompt context.
- `RAGContext`: Assembled context including system instructions and character metrics.
- `RAGResponse`: Pipeline output containing prediction, status, evidence provenance, and latencies.

### Service Layer (`app/services/rag_service.py`)
- `build_rag_context(fused_results, question, task_type, max_context_chars)`: Context builder.
- `RAGPipelineService`: Orchestrator class.

---

## 4. Execution Commands

### CLI Pipeline Runner
```bash
cd Backend
venv/bin/python scripts/run_rag_pipeline.py --question "Assess building damage after tornado"
```

### Run Unit Tests
```bash
cd Backend
venv/bin/pytest tests/unit/test_rag_service.py -v
```
