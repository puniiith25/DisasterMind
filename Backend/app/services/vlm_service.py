"""
Qwen2.5-VL Vision-Language Model Service for Disaster Reasoning.

Features:
- Configurable model identifier and execution device (MPS / CUDA / CPU)
- Lazy loading and instance caching (does not reload model on every query)
- Memory-aware visual resolution budgeting for 8 GB RAM constraints
- Explicit handling of bi-temporal (pre/post) imagery, single-image, and text prompts
- Execution timing, latency tracking, and robust error recovery
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
import torch

from app.schemas.vlm import VLMInferenceRequest, VLMInferenceResponse

import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

logger = logging.getLogger(__name__)


def select_device(preferred: str = "auto") -> str:
    """Determine the optimal execution device."""
    pref = preferred.lower().strip()
    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class QwenVLMService:
    """Reusable, memory-aware inference service for Qwen2.5-VL."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "auto",
        torch_dtype: Optional[torch.dtype] = None,
        min_pixels: int = 128 * 128,
        max_pixels: int = 384 * 384,
    ) -> None:
        self.model_id = model_id
        self.device = select_device(device)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Select appropriate precision
        if torch_dtype is not None:
            self.torch_dtype = torch_dtype
        elif self.device in ("cuda", "mps"):
            self.torch_dtype = torch.float16
        else:
            self.torch_dtype = torch.float32

        self.model: Any = None
        self.processor: Any = None
        self.is_loaded: bool = False
        self.load_time_seconds: float = 0.0

    def load_model(self) -> None:
        """Load model and processor into memory if not already initialized."""
        if self.is_loaded and self.model is not None and self.processor is not None:
            return

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        start_time = time.perf_counter()
        logger.info(f"Loading Qwen2.5-VL from '{self.model_id}' onto {self.device}...")

        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.eval()
        except Exception as load_err:
            if self.device != "cpu":
                logger.warning(f"Could not load onto {self.device} ({load_err}). Falling back to CPU for real inference...")
                self.device = "cpu"
                self.torch_dtype = torch.float32
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                ).to("cpu")
                self.model.eval()
            else:
                raise load_err

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        self.load_time_seconds = time.perf_counter() - start_time
        self.is_loaded = True
        logger.info(f"Qwen2.5-VL loaded successfully in {self.load_time_seconds:.2f}s")

    def unload_model(self) -> None:
        """Release model weights and clear accelerator cache."""
        self.model = None
        self.processor = None
        self.is_loaded = False
        if self.device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif self.device == "cuda" and hasattr(torch, "cuda"):
            torch.cuda.empty_cache()

    def _prepare_inputs(
        self,
        question: str,
        pre_image_path: Optional[str] = None,
        post_image_path: Optional[str] = None,
        image_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Image.Image]]:
        """Validate paths, load PIL images, and format chat template messages."""
        content: List[Dict[str, Any]] = []
        loaded_images: List[Image.Image] = []

        # Multi-image case (2 or more photos / drone captures / satellite views)
        if image_paths and len(image_paths) >= 2:
            for p in image_paths:
                path_obj = Path(p).resolve()
                if not path_obj.exists():
                    logger.warning(f"Image not found: {p}")
                    continue
                try:
                    img = Image.open(path_obj).convert("RGB")
                    loaded_images.append(img)
                    content.append({"type": "image", "image": img})
                except Exception as err:
                    logger.warning(f"Error loading image {p}: {err}")

            instruction = (
                f"Examine all {len(loaded_images)} disaster scene images carefully and analyze visual changes, damage, and evidence.\n\n"
                f"Question: {question.strip()}"
            )
            content.append({"type": "text", "text": instruction})

        elif pre_image_path and post_image_path:
            # Bi-temporal case
            for label, p in [("pre-disaster", pre_image_path), ("post-disaster", post_image_path)]:
                path_obj = Path(p).resolve()
                if not path_obj.exists():
                    raise FileNotFoundError(f"{label.capitalize()} image not found: {p}")
                img = Image.open(path_obj).convert("RGB")
                loaded_images.append(img)
                content.append({"type": "image", "image": img})

            instruction = (
                "Compare the pre-disaster and post-disaster images and answer the question briefly.\n\n"
                f"Question: {question.strip()}"
            )
            content.append({"type": "text", "text": instruction})

        elif image_path or (image_paths and len(image_paths) == 1) or post_image_path or pre_image_path:
            # Single image case
            single_p = image_path or (image_paths[0] if image_paths else None) or post_image_path or pre_image_path
            path_obj = Path(single_p).resolve()  # type: ignore[arg-type]
            if not path_obj.exists():
                raise FileNotFoundError(f"Image not found: {single_p}")
            img = Image.open(path_obj).convert("RGB")
            loaded_images.append(img)
            content.append({"type": "image", "image": img})

            instruction = (
                "Examine the disaster image and answer the question briefly.\n\n"
                f"Question: {question.strip()}"
            )
            content.append({"type": "text", "text": instruction})

        else:
            # Text-only disaster reasoning
            content.append({
                "type": "text",
                "text": f"Answer the following disaster-related question briefly.\n\nQuestion: {question.strip()}",
            })

        messages = [{"role": "user", "content": content}]
        return messages, loaded_images

    def generate(
        self,
        request: VLMInferenceRequest,
    ) -> VLMInferenceResponse:
        """Run end-to-end inference with structured error handling and latency tracking."""
        start_time = time.perf_counter()

        try:
            self.load_model()

            messages, loaded_images = self._prepare_inputs(
                question=request.question,
                pre_image_path=request.pre_image_path,
                post_image_path=request.post_image_path,
                image_path=request.image_path,
                image_paths=request.image_paths,
            )

            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            processor_kwargs: Dict[str, Any] = {
                "text": [prompt_text],
                "padding": True,
                "return_tensors": "pt",
            }
            if loaded_images:
                processor_kwargs["images"] = loaded_images

            inputs = self.processor(**processor_kwargs)
            inputs = {
                k: v.to(self.device, dtype=self.torch_dtype)
                if (hasattr(v, "to") and torch.is_floating_point(v))
                else (v.to(self.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    do_sample=(request.temperature > 0.0),
                    temperature=(request.temperature if request.temperature > 0.0 else None),
                    use_cache=True,
                )

            # Strip input tokens from generation
            input_length = inputs["input_ids"].shape[1]
            generated_ids = generated_ids[:, input_length:]

            decoded = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            prediction = decoded[0].strip()
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            return VLMInferenceResponse(
                prediction=prediction,
                latency_ms=latency_ms,
                device=self.device,
                model_id=self.model_id,
                status="SUCCESS",
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"VLM inference failed: {exc}", exc_info=True)
            return VLMInferenceResponse(
                prediction="",
                latency_ms=latency_ms,
                device=self.device,
                model_id=self.model_id,
                status="ERROR",
                error_message=str(exc),
            )


# Global singleton instance for reuse across requests
_VLM_SERVICE_INSTANCE: Optional[QwenVLMService] = None


def get_vlm_service(
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    device: str = "auto",
) -> QwenVLMService:
    """Retrieve or create the shared QwenVLMService instance."""
    global _VLM_SERVICE_INSTANCE
    if _VLM_SERVICE_INSTANCE is None:
        _VLM_SERVICE_INSTANCE = QwenVLMService(model_id=model_id, device=device)
    return _VLM_SERVICE_INSTANCE
