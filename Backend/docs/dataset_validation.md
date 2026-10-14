# Topic 01: DisasterM3 Dataset Validation & Partitioning

## 1. Objective
To establish a typed, robust, and non-destructive validation engine for the DisasterM3 benchmark dataset (`disasterm3_qa.json`), ensuring:
- Strict Pydantic-based schema conformance.
- Complete question and ground truth answer integrity.
- Verified physical image file existence on disk (supporting both bi-temporal pre/post images and single images).
- Duplicate ID and duplicate record detection.
- Stratified, deterministic train/dev/test partitioning without data leakage.
- Non-destructive reporting (preserving original dataset content and SHA-256 hashes).

---

## 2. Architecture & Schemas

### Data Model (`app/schemas/dataset.py`)
```python
class DisasterM3Record(BaseModel):
    id: str                                    # Unique ID: dm3_XXXXX
    task: str                                  # One of 6 disaster evaluation tasks
    question: str                              # Question prompt (non-empty)
    answer: str                                # Ground truth answer (non-empty)
    image_type: Optional[str]                  # Modality ("Optical", "SAR")
    pre_image: Optional[str]                   # Pre-disaster image path
    post_image: Optional[str]                  # Post-disaster image path
    image: Optional[str]                       # Single image path
    options: Optional[List[Union[str, int, float]]]  # MCQ options
    answer_option: Optional[str]               # Option letter (e.g. "A", "C, D")
    description: Optional[str]                 # Context/metadata
```

### Validation Service (`app/services/dataset_validator.py`)
- **`DatasetValidator`**:
  - `validate_record`: Individual record validator enforcing field presence, type safety, and disk path resolution.
  - `validate_dataset`: Comprehensive batch validator generating structured `ValidationReport`.
  - `split_dataset`: Deterministic partitioner stratified by task category with strict disjoint assertion ($Train \cap Dev \cap Test = \emptyset$).
  - `save_report`: Persists findings to `Backend/data/reports/validation_report.json`.
  - `save_splits`: Generates split JSONs into `Backend/data/disasterm3/splits/`.

---

## 3. Files Added / Modified

- `app/schemas/dataset.py`: Strongly typed Pydantic models for DisasterM3 records, issues, and reports.
- `app/schemas/__init__.py`: Package export for dataset schemas.
- `app/services/dataset_validator.py`: Core dataset validation and splitting service.
- `app/services/__init__.py`: Package export for services.
- `scripts/validate_dataset.py`: Command-line interface with reporting and splitting flags.
- `tests/unit/test_dataset_validation.py`: 10 comprehensive unit tests.
- `docs/dataset_validation.md`: This documentation.
- `data/reports/validation_report.json`: Validation report for the 1,599 DisasterM3 records.
- `data/disasterm3/splits/`: Stratified train (1,119), dev (240), and test (240) sets.
- `.gitignore`: Root Git ignore protection.

---

## 4. Installation Requirements

Dependencies are managed in `Backend/venv`:
```bash
source Backend/venv/bin/activate
pip install pydantic pytest
```

---

## 5. Commands to Execute

### Run Dataset Validation CLI
```bash
cd Backend
venv/bin/python scripts/validate_dataset.py --dataset data/disasterm3/disasterm3_qa.json --split
```

### Run Automated Pytest Suite
```bash
cd Backend
venv/bin/pytest tests/unit/test_dataset_validation.py -v
```

---

## 6. Actual Validation Metrics

Execution against `Backend/data/disasterm3/disasterm3_qa.json`:
* **Total Records**: 1,599
* **Valid Records**: 1,599 (100%)
* **Invalid Records**: 0
* **Duplicate IDs**: 0
* **Missing Images**: 0 (all pre, post, and single image files exist on disk)
* **Summary Status**: **PASS**

### Task Distribution:
* `Building Damage Counting`: 300
* `Road Damage Counting`: 300
* `Disaster Scene Recognition`: 300
* `Disaster Bearing Bodies Recognition`: 299
* `Disaster Type Recognition`: 200
* `Relational Reasoning`: 200

### Modality Distribution:
* `Optical`: 1,375
* `SAR`: 224

### Stratified Splits:
* **Train**: 1,119 records (70%)
* **Dev**: 240 records (15%)
* **Test**: 240 records (15%)

---

## 7. Test Results
10/10 automated tests passed:
- `test_valid_records_parsing`: **PASS**
- `test_missing_question_error`: **PASS**
- `test_missing_answer_error`: **PASS**
- `test_duplicate_id_detection`: **PASS**
- `test_missing_image_file_reported`: **PASS**
- `test_missing_image_reference`: **PASS**
- `test_invalid_item_type`: **PASS**
- `test_stratified_splits_reproducibility_and_zero_leakage`: **PASS**
- `test_save_splits_and_reports`: **PASS**
- `test_source_dataset_file_unmodified`: **PASS** (SHA-256 verified)

---

## 8. Known Limitations
- The dataset paths in `disasterm3_qa.json` are formatted as relative paths starting with `data/disasterm3/...`. Execution from outside `Backend/` requires passing the appropriate `--base-dir` parameter.

---

## 9. Next Step
Proceed to **Topic 02 — Qwen Vision-Language Baseline** (`feature/dm-02-vlm-baseline`).
