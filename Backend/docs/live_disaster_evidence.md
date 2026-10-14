# Topic 09: Live Disaster Evidence Ingestion

## 1. Objective
To dynamically ingest real-time and external disaster bulletins (e.g., USGS Earthquake Hazards Program, ReliefWeb, GDACS) and convert them into standardized `EvidenceRecord`s for vector indexing and live RAG retrieval.

---

## 2. Architecture & Normalization

```
External Live Feeds (USGS / ReliefWeb / GDACS GeoJSON APIs)
                      │
                      ▼
         LiveFeedService.fetch_usgs_feed()
                      │
                      ▼
            LiveDisasterFeedItem
   (event_id, title, coordinates, severity, summary)
                      │
                      ▼
       LiveFeedService.convert_to_evidence()
                      │
                      ▼
            EvidenceRecord
   (deterministic evi_live_{hash}, provenance: EXTERNAL_REPORT)
                      │
                      ▼
           Upserted to Qdrant Index
```

---

## 3. Schemas & Service

### Data Models (`app/schemas/live_feed.py`)
- `LiveDisasterFeedItem`: Structured live incident payload.
- `LiveFeedIngestRequest`: Request options (feed source, magnitude filter, limit, auto_index).
- `LiveFeedIngestResponse`: Summary of items ingested and indexed.

### Service Layer (`app/services/live_feed_service.py`)
- `LiveFeedService`: Ingestion engine with resilient offline fallbacks.

---

## 4. Verification

```bash
cd Backend
venv/bin/pytest tests/unit/test_live_feed_service.py -v
```
All 5 automated tests pass.
