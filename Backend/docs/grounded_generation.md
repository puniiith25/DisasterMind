# Topic 08: Grounded Generation & Citation Engine

## 1. Objective
To guarantee that DisasterMind's predictions are factual, auditable, and traceable to retrieved evidence snippets.
Key components:
- Sentence-level claim decomposition.
- Lexical & semantic overlap scoring against evidence passages.
- Automated citation synthesis (`[cit_1: evi_xxx]`).
- Hallucination risk grading (`LOW`, `MEDIUM`, `HIGH`).

---

## 2. Architecture & Schemas

### Data Models (`app/schemas/grounding.py`)
- `Citation`: Individual citation linking a claim to an evidence record with snippet preview.
- `GroundedClaim`: Decomposed atomic statement with boolean support status and confidence score.
- `GroundedPrediction`: Complete grounded prediction text, citation index, claims audit, and hallucination risk rating.

### Service Layer (`app/services/grounding_service.py`)
- `GroundingService`:
  - `decompose_claims(text)`
  - `compute_overlap_score(claim, evidence_text)`
  - `evaluate(prediction, evidence_snippets)`

---

## 3. Testing & Verification

```bash
cd Backend
venv/bin/pytest tests/unit/test_grounding_service.py -v
```
All 5 automated tests pass.
