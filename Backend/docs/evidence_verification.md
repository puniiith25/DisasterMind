# Topic 10: Evidence Verification & Contradiction Detection

## 1. Objective
To prevent catastrophic hallucination in disaster relief planning by detecting factual contradictions between model predictions and verified evidence.
Categories:
- `SUPPORTED`: Direct factual alignment with retrieved satellite evidence.
- `CONTRADICTED`: Conflicting damage counts, mismatched disaster events, or polarity/negation reversals.
- `UNVERIFIED`: Statements with insufficient retrieval overlap.

---

## 2. Architecture & Schemas

### Data Models (`app/schemas/verification.py`)
- `VerificationVerdict`: Enum (`SUPPORTED`, `CONTRADICTED`, `UNVERIFIED`).
- `ClaimVerificationResult`: Detailed audit per sentence claim with confidence and explanation.
- `VerificationReport`: Aggregate veracity metrics and critical contradiction flags.

### Service Layer (`app/services/verification_service.py`)
- `EvidenceVerificationService`:
  - `extract_numbers(text)`
  - `extract_disasters(text)`
  - `check_claim_against_snippet(claim, snippet)`
  - `verify(prediction, evidence_snippets)`

---

## 3. Testing & Verification

```bash
cd Backend
venv/bin/pytest tests/unit/test_verification_service.py -v
```
All 5 automated tests pass.
