# DisasterMind: Multimodal RAG for Disaster Assessment

DisasterMind is an end-to-end multimodal Retrieval-Augmented Generation (RAG) platform tailored for disaster assessment and satellite/aerial imagery analysis, built upon the **DisasterM3** benchmark.

---

## 🏛️ System Architecture

```
User Query (Question + Optional Pre/Post Images)
       │
       ▼
  FastAPI REST API (`app/main.py`)
       │
       ▼
 RAG Pipeline Service (`app/services/rag_service.py`)
       │
       ├── Multimodal Retrieval Service (`app/services/multimodal_service.py`)
       │     ├── Text Query  ──► BGE-M3 Dense Embedding ──► Local Qdrant Index
       │     └── Image Query ──► OpenAI CLIP-ViT         ──► Visual Search
       │                                │
       │                                ▼
       │                    Reciprocal Rank Fusion (RRF)
       │
       ├── Context Augmentation Engine (Character & Token Budget Enforcement)
       │
       └── Qwen2.5-VL Vision-Language Model (`app/services/vlm_service.py`)
             └── Grounded Disaster Prediction with Provenance Audit Trail
```

---

## 🚀 Quick Start

### 1. Setup Virtual Environment
```bash
cd Backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # or: pip install fastapi uvicorn pydantic pytest sentence-transformers qdrant-client torch
```

### 2. Run API Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Interactive API docs are available at **http://localhost:8000/docs**.

### 3. Run Automated Unit Tests
```bash
pytest tests/unit -v
```

---

## 🛠️ CLI Tools & Workflow

| Script | Purpose | Command |
|---|---|---|
| `validate_dataset.py` | Validates records and builds stratified splits | `python scripts/validate_dataset.py --split` |
| `evaluate_vlm_baseline.py` | Runs Qwen2.5-VL baseline benchmark | `python scripts/evaluate_vlm_baseline.py --samples 5` |
| `build_evidence_corpus.py` | Extracts standardized evidence corpus | `python scripts/build_evidence_corpus.py` |
| `generate_embeddings.py` | BGE-M3 dense embeddings with SQLite cache | `python scripts/generate_embeddings.py --limit 100` |
| `build_qdrant_index.py` | Indexes corpus into local disk Qdrant | `python scripts/build_qdrant_index.py` |
| `run_multimodal_retrieval.py` | Smoke-tests multimodal RRF search | `python scripts/run_multimodal_retrieval.py --question "Damage assessment"` |
| `run_rag_pipeline.py` | Runs full end-to-end RAG prediction | `python scripts/run_rag_pipeline.py --question "How many buildings collapsed?"` |

---

## 📑 Feature Modules & Documentation

- [DM-01: Dataset Validation & Partitioning](docs/dataset_validation.md)
- [DM-02: Qwen2.5-VL Baseline Inference](docs/vlm_baseline.md)
- [DM-03: Multimodal Evidence Corpus](docs/evidence_corpus.md)
- [DM-04: BGE-M3 Dense Embeddings](docs/bge_embeddings.md)
- [DM-05: Local Qdrant Vector Indexing](docs/qdrant_indexing.md)
- [DM-06: Multimodal RRF Retrieval](docs/multimodal_retrieval.md)
- [DM-07: Multimodal RAG Pipeline](docs/rag_pipeline.md)
- [DM-08: FastAPI REST API](docs/fastapi_server.md)
