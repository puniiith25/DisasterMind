# Topic 02: Qwen2.5-VL Vision-Language Baseline

## 1. Objective
To establish a memory-aware, modular, and reproducible inference service for `Qwen2.5-VL-3B-Instruct`.
The module handles:
- Bi-temporal disaster satellite/aerial imagery (`pre_image` and `post_image`).
- Single-image disaster scene recognition.
- Text-only disaster domain reasoning.
- Automatic device selection (Apple MPS, CUDA, CPU).
- Bounded visual resolution (`224*224` to `448*448`) suitable for local 8 GB RAM machines.
- Latency and peak memory measurement.
- Comprehensive error handling with non-crashing responses.

---

## 2. Architecture & Service Design

### Schema Layer (`app/schemas/vlm.py`)
- `VLMInferenceRequest`: Typed prompt payload with optional pre/post/single image paths, token limits, and temperature.
- `VLMInferenceResponse`: Structured prediction with latency in milliseconds, execution device, model ID, and status (`SUCCESS` / `ERROR`).
- `VLMBenchmarkReport`: Standardized output schema for reproducible baseline benchmarks.

### Service Layer (`app/services/vlm_service.py`)
- `QwenVLMService`:
  - **Singleton / Instance Reuse**: Reuses model weights across calls to avoid reloading large weights into memory.
  - **Precision Alignment**: Automatically matches image feature tensors to `torch.float16` for MPS hardware efficiency.
  - **Chat Template Formatting**: Constructs domain-specific prompts directing the model to compare bi-temporal features or examine single disaster scenes.
  - **Memory Cleanup**: Frees accelerator cache via `torch.mps.empty_cache()` upon model unload.

---

## 3. Files Added / Modified

- `app/schemas/vlm.py`: Pydantic models for VLM inference requests, responses, and benchmark reports.
- `app/services/vlm_service.py`: Reusable, memory-aware Qwen2.5-VL service.
- `scripts/evaluate_vlm_baseline.py`: CLI benchmarking tool for measuring latency and memory.
- `tests/unit/test_vlm_service.py`: Automated unit test suite with mocked inference.
- `docs/vlm_baseline.md`: This documentation.

---

## 4. Execution Commands

### Run Unit Tests
```bash
cd Backend
venv/bin/pytest tests/unit/test_vlm_service.py -v
```

### Run Baseline Benchmark CLI
```bash
cd Backend
venv/bin/python scripts/evaluate_vlm_baseline.py --samples 5 --device mps --output-report data/reports/vlm_baseline_evaluation.json
```

---

## 5. Performance Characteristics (Apple M1, 8 GB RAM)

- **Model Loading Time**: ~22-30 seconds to load weights onto Apple MPS.
- **Precision**: `float16` on MPS / CUDA; `float32` fallback on CPU.
- **Resolution Bounds**: `min_pixels = 224 * 224`, `max_pixels = 448 * 448` to limit visual tokens.
- **Memory Footprint**: ~2.5 - 3.2 GB Unified Memory during generation.

---

## 6. Verification Status

- 9/9 unit tests passed:
  - `test_select_device_explicit`: **PASS**
  - `test_select_device_auto`: **PASS**
  - `test_prepare_inputs_bitemporal`: **PASS**
  - `test_prepare_inputs_single_image`: **PASS**
  - `test_prepare_inputs_text_only`: **PASS**
  - `test_missing_image_file_returns_error`: **PASS**
  - `test_mocked_generate_success`: **PASS**
  - `test_mocked_generate_exception_recovery`: **PASS**
  - `test_singleton_vlm_service`: **PASS**
- Combined test suite (Topics 01 & 02): 19/19 tests passed.

---

## 7. Next Step
Proceed to **Topic 03 — Evidence Corpus** (`feature/dm-03-evidence-corpus`).
