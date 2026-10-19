"""
CLIP Image Embedding Service for DisasterMind.

Features:
- Lazy model loading (openai/clip-vit-base-patch32)
- MPS / CUDA / CPU device auto-selection
- L2-normalized 512-dim visual CLIP embeddings
- Supports PIL Image objects and file paths
- Finite-value validation
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def select_clip_device(preferred: str = "auto") -> str:
    pref = preferred.lower().strip()
    if pref not in ("auto", ""):
        return pref
    if torch.cuda.is_available():
        return "cuda"
    # CPU is fast (<20ms) for CLIP-ViT and avoids Apple Silicon MPS synchronization deadlocks
    return "cpu"


class CLIPImageService:
    """
    Lazy-loading CLIP image embedding service.

    Produces 512-dim L2-normalized visual embeddings for disaster scene images.
    Compatible with openai/clip-vit-base-patch32 via HuggingFace Transformers.
    """

    DEFAULT_MODEL = "openai/clip-vit-base-patch32"
    DIMENSION = 512

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.device = select_clip_device(device)
        self.dimension = self.DIMENSION

        self.model: Any = None
        self.processor: Any = None
        self.is_loaded: bool = False
        self.load_time_seconds: float = 0.0

    def load_model(self) -> None:
        """Load CLIP model and processor if not already loaded."""
        if self.is_loaded:
            return

        from transformers import CLIPModel, CLIPProcessor

        start = time.perf_counter()
        logger.info(f"Loading {self.model_id} on {self.device} ...")

        try:
            self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_id)
        except Exception as e:
            if self.device != "cpu":
                logger.warning(f"Failed on {self.device}, falling back to CPU: {e}")
                self.device = "cpu"
                self.model = CLIPModel.from_pretrained(self.model_id).to("cpu")
                self.processor = CLIPProcessor.from_pretrained(self.model_id)
            else:
                raise

        self.model.eval()
        self.load_time_seconds = time.perf_counter() - start
        self.is_loaded = True
        logger.info(f"Loaded {self.model_id} in {self.load_time_seconds:.2f}s")

    def _validate_vector(self, vec: List[float]) -> None:
        if len(vec) != self.dimension:
            raise ValueError(
                f"CLIP vector dimension mismatch: expected {self.dimension}, got {len(vec)}"
            )
        arr = np.array(vec, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Non-finite values detected in CLIP embedding")

    def embed_image(self, image: Image.Image | str | Path) -> List[float]:
        """
        Compute a 512-dim CLIP visual embedding for a single image.

        Args:
            image: PIL Image, file path string, or Path object

        Returns:
            List[float] of dimension 512, L2-normalized
        """
        self.load_model()

        if not isinstance(image, Image.Image):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            image = Image.open(path).convert("RGB")
        else:
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            vision_out = self.model.vision_model(
                pixel_values=inputs["pixel_values"]
            )
            pooled = vision_out.pooler_output
            features = self.model.visual_projection(pooled)

        # L2-normalize
        features = features / features.norm(dim=-1, keepdim=True)
        vec = features[0].cpu().tolist()
        self._validate_vector(vec)
        return vec

    def embed_text_query(self, text: str) -> List[float]:
        """
        Compute a 512-dim CLIP text embedding for cross-modal text-to-image retrieval.

        Args:
            text: Natural-language query string

        Returns:
            List[float] of dimension 512, L2-normalized
        """
        self.load_model()

        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            text_out = self.model.text_model(**inputs)
            pooled = text_out.pooler_output
            features = self.model.text_projection(pooled)

        features = features / features.norm(dim=-1, keepdim=True)
        vec = features[0].cpu().tolist()
        self._validate_vector(vec)
        return vec


# Module-level singleton
_CLIP_SERVICE_INSTANCE: Optional[CLIPImageService] = None


def get_clip_service(device: str = "auto") -> CLIPImageService:
    global _CLIP_SERVICE_INSTANCE
    if _CLIP_SERVICE_INSTANCE is None:
        _CLIP_SERVICE_INSTANCE = CLIPImageService(device=device)
    return _CLIP_SERVICE_INSTANCE
