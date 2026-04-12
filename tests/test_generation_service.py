"""Tests for backend generation orchestration."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from PIL import Image

from backend.app.generation_service import GenerationUnavailableError, generate_renovation


class GenerationServiceTests(unittest.TestCase):
    """Verify generation service behavior without running a real model."""

    def test_generate_renovation_logs_and_fails_when_real_generator_is_unavailable(self) -> None:
        """The service should log the real-generation attempt and fail instead of mocking output."""

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "room.png"
            Image.new("RGB", (256, 192), color="white").save(image_path)

            with patch(
                "backend.app.generation_service.ControlNetSDXLGenerator.generate",
                side_effect=RuntimeError("CUDA not available"),
            ):
                with self.assertLogs("backend.app.generation_service", level="INFO") as captured_logs:
                    with self.assertRaises(GenerationUnavailableError):
                        generate_renovation(image_path, "Living Room", "Modern Luxury")

        log_output = "\n".join(captured_logs.output)
        self.assertIn("Entering generation", log_output)
        self.assertIn("Mock mode disabled", log_output)
        self.assertIn("Chosen model IDs", log_output)

    def test_generate_renovation_logs_output_path_when_generation_succeeds(self) -> None:
        """The service should log the generated output path on successful real generation."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "room.png"
            output_path = temp_path / "generated.png"
            Image.new("RGB", (256, 192), color="white").save(image_path)
            Image.new("RGB", (256, 192), color="gray").save(output_path)

            fake_generation_result = SimpleNamespace(
                room_type="Living Room",
                style="Modern Luxury",
                output_image_path=output_path,
            )

            with patch(
                "backend.app.generation_service.ControlNetSDXLGenerator.generate",
                return_value=fake_generation_result,
            ):
                with self.assertLogs("backend.app.generation_service", level="INFO") as captured_logs:
                    result = generate_renovation(image_path, "Living Room", "Modern Luxury")

        self.assertEqual(result.generation_mode, "real")
        self.assertEqual(result.room_type, "Living Room")
        self.assertEqual(result.style, "Modern Luxury")
        self.assertEqual(result.artifacts.output_image_path, output_path)
        self.assertIn(f"Output file path: {output_path}", "\n".join(captured_logs.output))

    def test_generate_renovation_preserves_blank_output_error_message(self) -> None:
        """Blank-output failures should remain visible to the caller."""

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "room.png"
            Image.new("RGB", (256, 192), color="white").save(image_path)

            with patch(
                "backend.app.generation_service.ControlNetSDXLGenerator.generate",
                side_effect=RuntimeError("Generated image appears blank or nearly black."),
            ):
                with self.assertRaisesRegex(
                    GenerationUnavailableError,
                    "Generated image appears blank or nearly black",
                ):
                    generate_renovation(image_path, "Living Room", "Modern Luxury")
