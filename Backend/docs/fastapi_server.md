# Topic 08: DisasterMind FastAPI REST API

## 1. Objective
To expose the complete DisasterMind multimodal RAG system via a production-grade FastAPI REST API with Swagger/OpenAPI documentation, CORS middleware, and dependency injection.

---

## 2. API Endpoints

| Method | Path | Summary | Tags |
|---|---|---|---|
| `GET` | `/` | API Root / Endpoint Directory | General |
| `GET` | `/health` | Health Check (Vector Index Status) | General |
| `POST` | `/api/v1/rag/predict` | End-to-End Multimodal RAG Prediction | RAG Pipeline |
| `POST` | `/api/v1/retrieval/multimodal` | Multimodal RRF Retrieval | Retrieval |
| `POST` | `/api/v1/retrieval/dense` | Dense Text Similarity Search | Retrieval |
| `POST` | `/api/v1/vlm/generate` | Direct VLM Baseline Generation | VLM Baseline |
| `GET` | `/api/v1/dataset/stats` | Dataset & Corpus Statistics | Dataset |
| `GET` | `/docs` | Interactive Swagger UI | Documentation |
| `GET` | `/redoc` | Interactive ReDoc UI | Documentation |

---

## 3. Running the Server

```bash
cd Backend
venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive Swagger documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 4. Automated Testing

All endpoints are fully covered with dependency-mocked unit tests:
```bash
cd Backend
venv/bin/pytest tests/unit/test_api.py -v
```
