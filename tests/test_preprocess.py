"""Tests for image preprocessing."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from app.pipeline.preprocess import preprocess_image


class PreprocessTests(unittest.TestCase):
    """Verify preprocessing artifacts are written to disk."""

    def test_preprocess_image_writes_edge_map_and_metadata(self) -> None:
        """Preprocessing should persist an edge map and report resized dimensions."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "room.png"

            image = Image.new("RGB", (640, 480), color="white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((80, 80, 560, 400), outline="black", width=6)
            image.save(image_path)

            result = preprocess_image(image_path=image_path, max_dimension=256, temp_root=temp_path)

            self.assertEqual(result.source_path, image_path.resolve())
            self.assertTrue(result.temp_dir.exists())
            self.assertTrue(result.edge_map_path.exists())
            self.assertEqual(result.edge_map_path.parent, result.temp_dir)
            self.assertLessEqual(result.metadata["resized_width"], 256)
            self.assertLessEqual(result.metadata["resized_height"], 256)
