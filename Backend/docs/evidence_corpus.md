# Topic 03: Multimodal Evidence Corpus

## 1. Objective
To build a deterministic, standardized evidence corpus generator that converts raw QA and benchmark records into structured multimodal evidence units (`EvidenceRecord`).
Key guarantees:
- Provenance separation: explicitly separates dataset QA pairs (`dataset_qa`), descriptive scene contexts (`verified_description`), external incident reports (`external_report`), and model outputs (`model_output`).
- Explicit visual asset tracking: categorizes visual evidence as `pre_disaster`, `post_disaster`, or `observation`, with modality flags (`Optical`, `SAR`).
- Verified metadata extraction: parses disaster event names (e.g. Joplin Tornado, Libya Flood) from verified records without hallucinating or fabricating timestamps or claims.
- Collision-resistant stable identifiers (`evi_{hash}`).
- Deterministic deduplication and non-destructive execution.

---

## 2. Architecture & Schemas

### Data Model (`app/schemas/evidence.py`)
```python
class EvidenceRecord(BaseModel):
    evidence_id: str                      # Deterministic content hash ID
    record_id: Optional[str]              # Originating record ID (e.g. dm3_00000)
    text: str                             # Normalized searchable text
    task_type: Optional[str]              # Disaster task category
    question: Optional[str]               # Question prompt if QA
    answer: Optional[str]                 # Historical answer if QA
    images: List[EvidenceImageReference]  # Linked visual assets with roles
    disaster_type: Optional[str]          # Hurricane, Tornado, Earthquake, etc.
    location: Optional[str]               # Event name / location if verified
    source_type: EvidenceSourceType       # dataset_qa / verified_description
    source_reference: str                 # Dataset origin or report source
    observed_at: Optional[str]            # ISO timestamp if known; None if unverified
    published_at: Optional[str]           # ISO timestamp if known; None if unverified
    metadata: Dict[str, Any]              # Modality, options, and auxiliary metadata
```

### Service Layer (`app/services/corpus_service.py`)
- `EvidenceCorpusGenerator`:
  - `create_evidence_from_qa_record`: Converts 1,599 QA records into verified multimodal units.
  - `create_evidence_from_retrieval_item`: Converts background corpus into verified descriptive evidence.
  - `build_corpus`: Deduplicates content across inputs and aggregates corpus statistics.
  - `save_corpus`: Persists standardized records to `data/disasterm3/retrieval/standardized_corpus.json`.
  - `save_statistics`: Persists summary to `data/reports/corpus_statistics.json`.

---

## 3. Files Added / Modified

- `app/schemas/evidence.py`: Pydantic models for evidence items, visual references, and corpus statistics.
- `app/services/corpus_service.py`: Standardized evidence generation service.
- `scripts/build_evidence_corpus.py`: CLI tool for corpus construction and reporting.
- `tests/unit/test_evidence_corpus.py`: Automated unit tests covering hashing, deduplication, schema validation, and checksum integrity.
- `data/disasterm3/retrieval/standardized_corpus.json`: Generated standardized multimodal corpus (1,263 deduplicated records).
- `data/reports/corpus_statistics.json`: Corpus statistics report.
- `docs/evidence_corpus.md`: This documentation.

---

## 4. Execution Commands

### Generate Standardized Evidence Corpus
```bash
cd Backend
venv/bin/python scripts/build_evidence_corpus.py
```

### Run Unit Tests
```bash
cd Backend
venv/bin/pytest tests/unit/test_evidence_corpus.py -v
```

---

## 5. Corpus Statistics & Provenance Breakdown

- **Total Deduplicated Records**: 1,263
- **Provenance Distribution**:
  - `dataset_qa`: 1,079
  - `verified_description`: 184
- **Disaster Type Distribution**:
  - Earthquake: 326
  - Hurricane: 181
  - Flood: 150
  - Wildfire: 125
  - Tornado: 113
  - Tsunami: 83
  - Explosion: 51
  - Volcano: 43
  - Conflict: 6
  - Unspecified: 185
- **Referenced Visual Assets**: 1,959 images (1,660 Optical, 299 SAR)

---

## 6. Verification Status

- 6/6 tests passed in `test_evidence_corpus.py`.
- 25/25 tests passed across Topics 01, 02, and 03.
- Source dataset SHA-256 verified identical before and after corpus construction.

---

## 7. Next Step
Proceed to **Topic 04 — BGE-M3 Embeddings** (`feature/dm-04-bge-embeddings`).
