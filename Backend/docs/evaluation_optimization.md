
# Topic 13: Evaluation & Optimization Benchmark

## 1. Objective
To systematically evaluate DisasterMind's multimodal retrieval and generation quality against the DisasterM3 benchmark splits (`train`, `dev`, `test`).

---

## 2. Evaluation Dimensions

1. **Exact Match (EM)**: String and case-normalized accuracy for categorical and multiple-choice questions.
2. **Token F1**: Overlap of factual tokens for descriptive disaster scene assessments.
3. **Retrieval Provenance**: Rate of correct evidence inclusion in top-K fused list.
4. **End-to-End Latency**: Breakdowns across BGE-M3 embedding, Qdrant search, CLIP scoring, and VLM generation.

---

## 3. CLI Benchmark Runner

```bash
cd Backend
venv/bin/python scripts/evaluate_benchmark.py --split data/disasterm3/splits/dev.json --limit 50
```

Results are persisted to `Backend/data/reports/evaluation_benchmark_report.json`.

---

## 4. Verification

```bash
cd Backend
venv/bin/pytest tests/unit/test_evaluation_benchmark.py -v
```
All automated tests pass.
