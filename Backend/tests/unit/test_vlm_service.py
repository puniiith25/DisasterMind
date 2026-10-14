"""
Unit tests for QwenVLMService with mocked vision-language execution.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest
import torch

from app.schemas.vlm import VLMInferenceRequest
from app.services.vlm_service import QwenVLMService, get_vlm_service, select_device


@pytest.fixture
def dummy_images(tmp_path):
    """Create dummy test images."""
    pre_p = tmp_path / "pre.png"
    post_p = tmp_path / "post.png"
    single_p = tmp_path / "single.png"

    img = Image.new("RGB", (64, 64), color="blue")
    img.save(pre_p)
    img.save(post_p)
    img.save(single_p)

    return pre_p, post_p, single_p


class TestQwenVLMService:
    """Test suite for QwenVLMService logic and mocked inference."""

    def test_select_device_explicit(self):
        assert select_device("cpu") == "cpu"
        assert select_device("mps") == "mps"
        assert select_device("cuda") == "cuda"

    def test_select_device_auto(self):
        dev = select_device("auto")
        assert dev in ("mps", "cuda", "cpu")

    def test_prepare_inputs_bitemporal(self, dummy_images):
        pre_p, post_p, _ = dummy_images
        service = QwenVLMService(device="cpu")
        messages, loaded_images = service._prepare_inputs(
            question="What is damaged?",
            pre_image_path=str(pre_p),
            post_image_path=str(post_p),
        )

        assert len(loaded_images) == 2
        assert len(messages) == 1
        content = messages[0]["content"]
        assert len(content) == 3
        assert content[0]["type"] == "image"
        assert content[1]["type"] == "image"
        assert content[2]["type"] == "text"
        assert "Compare the pre-disaster and post-disaster images" in content[2]["text"]

    def test_prepare_inputs_single_image(self, dummy_images):
        _, _, single_p = dummy_images
        service = QwenVLMService(device="cpu")
        messages, loaded_images = service._prepare_inputs(
            question="Is there fire?",
            image_path=str(single_p),
        )

        assert len(loaded_images) == 1
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "image"
        assert "Examine the disaster image" in content[1]["text"]

    def test_prepare_inputs_text_only(self):
        service = QwenVLMService(device="cpu")
        messages, loaded_images = service._prepare_inputs(
            question="What precautions should be taken in a flood?",
        )

        assert len(loaded_images) == 0
        content = messages[0]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert "Answer the following disaster-related question" in content[0]["text"]

    def test_missing_image_file_returns_error(self, tmp_path):
        service = QwenVLMService(device="cpu")
        service.load_model = MagicMock()
        service.is_loaded = True
        service.model = MagicMock()
        service.processor = MagicMock()

        req = VLMInferenceRequest(
            question="Analyze damage",
            pre_image_path=str(tmp_path / "non_existent.png"),
            post_image_path=str(tmp_path / "another_missing.png"),
        )
        resp = service.generate(req)

        assert resp.status == "ERROR"
        assert "not found" in (resp.error_message or "").lower()
        assert resp.prediction == ""
        assert resp.latency_ms >= 0.0

    def test_mocked_generate_success(self, dummy_images):
        pre_p, post_p, _ = dummy_images
        service = QwenVLMService(device="cpu")
        service.load_model = MagicMock()
        service.is_loaded = True

        mock_model = MagicMock()
        mock_processor = MagicMock()

        # Mock processor behavior
        mock_processor.apply_chat_template.return_value = "<user>prompt"
        mock_inputs = {
            "input_ids": torch.tensor([[101, 102, 103]]),
            "pixel_values": torch.zeros((1, 3, 224, 224)),
        }
        mock_processor.return_value = mock_inputs

        # Model outputs [101, 102, 103, 201, 202] (2 generated tokens)
        mock_model.generate.return_value = torch.tensor([[101, 102, 103, 201, 202]])
        mock_processor.batch_decode.return_value = ["Buildings and roads are damaged."]

        service.model = mock_model
        service.processor = mock_processor

        req = VLMInferenceRequest(
            question="What is damaged?",
            pre_image_path=str(pre_p),
            post_image_path=str(post_p),
        )

        resp = service.generate(req)

        assert resp.status == "SUCCESS"
        assert resp.prediction == "Buildings and roads are damaged."
        assert resp.latency_ms > 0
        assert resp.model_id == "Qwen/Qwen2.5-VL-3B-Instruct"

    def test_mocked_generate_exception_recovery(self, dummy_images):
        _, _, single_p = dummy_images
        service = QwenVLMService(device="cpu")
        service.load_model = MagicMock()
        service.is_loaded = True

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "prompt"
        mock_processor.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
        }
        mock_model.generate.side_effect = RuntimeError("CUDA out of memory simulation")

        service.model = mock_model
        service.processor = mock_processor

        req = VLMInferenceRequest(
            question="Identify danger",
            image_path=str(single_p),
        )

        resp = service.generate(req)

        assert resp.status == "ERROR"
        assert "out of memory simulation" in (resp.error_message or "")
        assert resp.prediction == ""

    def test_singleton_vlm_service(self):
        s1 = get_vlm_service()
        s2 = get_vlm_service()
        assert s1 is s2
