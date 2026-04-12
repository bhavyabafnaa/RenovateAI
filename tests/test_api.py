"""Tests for FastAPI endpoints."""

from __future__ import annotations

from io import BytesIO
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app
from backend.app.generation_service import GenerationUnavailableError


def _make_test_image_bytes() -> bytes:
    """Create a small in-memory PNG for API tests."""

    image = Image.new("RGB", (128, 96), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ApiTests(unittest.TestCase):
    """Exercise lightweight API behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_endpoint_returns_status_and_styles(self) -> None:
        """The health endpoint should expose supported styles and runtime details."""

        runtime = {
            "configured_device": "cuda",
            "require_cuda": True,
            "torch_available": True,
            "torch_version": "test",
            "cuda_available": True,
            "cuda_version": "test",
            "cuda_device_count": 1,
            "cuda_device_name": "Test GPU",
            "selected_device": "cuda",
            "error": None,
        }

        with patch("backend.app.main.get_runtime_status", return_value=runtime):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "styles": ["modern", "scandinavian", "industrial"],
                "runtime": runtime,
            },
        )

    def test_health_reports_gpu_unavailable_when_cuda_is_required(self) -> None:
        """The health status should be explicit when required CUDA is missing."""

        runtime = {
            "configured_device": "cuda",
            "require_cuda": True,
            "torch_available": True,
            "torch_version": "test",
            "cuda_available": False,
            "cuda_version": None,
            "cuda_device_count": 0,
            "cuda_device_name": None,
            "selected_device": None,
            "error": None,
        }

        with patch("backend.app.main.get_runtime_status", return_value=runtime):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "gpu_unavailable")
        self.assertEqual(response.json()["runtime"], runtime)

    def test_generate_requires_uploaded_image(self) -> None:
        """The generate endpoint should reject requests without an image file."""

        response = self.client.post("/generate", data={"style": "modern"})

        self.assertEqual(response.status_code, 422)

    def test_generate_rejects_unsupported_style(self) -> None:
        """The generate endpoint should reject unsupported styles with a 400."""

        response = self.client.post(
            "/generate",
            files={"image": ("room.png", _make_test_image_bytes(), "image/png")},
            data={"style": "boho"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported style", response.json()["detail"])

    @patch("backend.app.main.generate_renovation")
    def test_generate_returns_503_when_real_generator_is_unavailable(self, mock_generate_renovation) -> None:
        """The generate endpoint should fail clearly when real generation is unavailable."""

        mock_generate_renovation.side_effect = GenerationUnavailableError("Real generator unavailable.")

        response = self.client.post(
            "/generate",
            files={"image": ("room.png", _make_test_image_bytes(), "image/png")},
            data={"style": "modern"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Real generator unavailable.")
