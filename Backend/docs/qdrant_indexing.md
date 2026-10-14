# DM-05: Qdrant Dense Text Indexing

## Overview

Topic DM-05 implements a production-grade local Qdrant vector index over the
DisasterM3 standardized evidence corpus using BGE-M3 1024-dim dense embeddings.
The index enables sub-millisecond ranked retrieval without any external server
— all data lives on disk as a local Qdrant directory.

## Architecture

```
standardized_corpus.json (1263 records)
         │
         ▼
 BGEM3EmbeddingService          ← DM-04: 1024-dim L2-normalized dense vectors
 (SQLite cache + MPS/CPU)
         │
         ▼
  QdrantIndexService             ← DM-05
  ├── build_index()              → embed + upsert all records into Qdrant
  ├── retrieve()                 → dense cosine similarity top-k search
  └── collection_info()          → collection health / point count
         │
         ▼
  Qdrant local disk             ← data/qdrant/
  (no Docker required)
```

## Key Components

### `app/schemas/retrieval.py`

| Class | Purpose |
|---|---|
| `CollectionStatus` | Enum: CREATED / UPDATED / EXISTS / DELETED |
| `IndexedPoint` | Point provenance (evidence_id, record_id, task_type) |
| `IndexBuildReport` | Full build metrics (count, latency, throughput, cache rate) |
| `RetrievalResult` | Single ranked search result with score and payload |
| `RetrievalResponse` | Ranked list + latency for a query |

### `app/services/qdrant_service.py`

| Method | Description |
|---|---|
| `build_index(corpus_path, batch_size, recreate)` | Load corpus → embed texts → upsert to Qdrant; idempotent |
| `retrieve(query, top_k)` | Embed query → cosine similarity search → ranked results |
| `collection_info()` | Returns existence status and point count |

**Qdrant API version**: qdrant-client 1.19.x  
- Uses `client.query_points(query=vector)` (replaces deprecated `client.search()`)  
- Uses `CollectionInfo.points_count` (replaces deprecated `vectors_count`)

### `scripts/build_qdrant_index.py`

CLI tool to build and smoke-test the index:

```bash
cd Backend/
python scripts/build_qdrant_index.py --limit 100 --batch-size 16
python scripts/build_qdrant_index.py  # full 1263 records
```

## Collection Schema

- **Collection name**: `disasterm3_text`
- **Vector size**: 1024 (BGE-M3 dense)
- **Distance**: Cosine
- **Storage**: `data/qdrant/` (local disk)

### Payload per point

```json
{
  "evidence_id": "evi_4a3e69afcd38",
  "record_id": "dm3_00000",
  "text": "Task: ... Question: ... Historical answer: ...",
  "task_type": "Disaster Bearing Bodies Recognition",
  "question": "...",
  "answer": "...",
  "disaster_type": "Earthquake",
  "location": "Turkey",
  "source_type": "qa",
  "images": [...]
}
```

## Test Coverage (9 tests — 9 passed)

| Test | What it verifies |
|---|---|
| `test_build_index_returns_report` | Report has correct indexed_count |
| `test_build_index_collection_created` | Collection created on disk with correct point count |
| `test_build_index_idempotent_rebuild` | Recreate clears old collection, reindex gives same count |
| `test_build_index_missing_corpus_raises` | FileNotFoundError on missing corpus |
| `test_retrieve_returns_ranked_results` | Returns RetrievalResponse with top_k ranked results |
| `test_retrieve_results_ranked_descending` | Scores are in descending order |
| `test_collection_info_before_build` | exists=False before index built |
| `test_collection_info_after_build` | exists=True with correct point count |
| `test_indexed_points_provenance` | Every evidence_id and record_id in report matches corpus |

All tests fully mocked — no real BGE-M3 model or Qdrant server needed.
