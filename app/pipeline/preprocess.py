"""Reusable image preprocessing helpers for the RenovateAI pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from uuid import uuid4

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessResult:
    """Structured output for a preprocessing run."""

    source_path: Path
    edge_map_path: Path
    temp_dir: Path
    metadata: dict[str, int | float | str]


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk in BGR format."""

    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {path}")

    return image


def resize_image(image: np.ndarray, minimum_long_side: int = 1024) -> tuple[np.ndarray, dict[str, int | float | str]]:
    """Upscale an image so its longest side is at least ``minimum_long_side``."""

    if minimum_long_side <= 0:
        raise ValueError("minimum_long_side must be greater than zero.")

    height, width = image.shape[:2]
    longest_side = max(height, width)
    scale = max(1.0, minimum_long_side / float(longest_side))

    if scale == 1.0:
        resized = image.copy()
    else:
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    resized_height, resized_width = resized.shape[:2]
    metadata = {
        "original_width": width,
        "original_height": height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "scale": scale,
        "minimum_long_side": minimum_long_side,
        "interpolation": "lanczos",
    }
    return resized, metadata


def generate_canny_edge_map(
    image: np.ndarray,
    low_threshold: int = 100,
    high_threshold: int = 200,
) -> np.ndarray:
    """Generate a Canny edge map from a BGR image."""

    if low_threshold < 0 or high_threshold < 0:
        raise ValueError("Canny thresholds must be non-negative.")
    if low_threshold >= high_threshold:
        raise ValueError("low_threshold must be smaller than high_threshold.")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    return cv2.Canny(blurred, low_threshold, high_threshold)


def save_edge_map(edge_map: np.ndarray, source_path: str | Path, temp_root: str | Path | None = None) -> tuple[Path, Path]:
    """Save an edge map to a dedicated temporary folder and return its paths."""

    root = Path(temp_root) if temp_root is not None else Path(tempfile.gettempdir())
    temp_dir = root / f"renovateai-preprocess-{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=False)

    source_name = Path(source_path).stem
    edge_map_path = temp_dir / f"{source_name}_edges.png"

    if not cv2.imwrite(str(edge_map_path), edge_map):
        raise ValueError(f"Failed to save edge map to {edge_map_path}")

    return edge_map_path, temp_dir


def preprocess_image(
    image_path: str | Path,
    minimum_long_side: int = 1024,
    low_threshold: int = 100,
    high_threshold: int = 200,
    temp_root: str | Path | None = None,
) -> PreprocessResult:
    """Load an image, resize it, generate a Canny edge map, and persist the result."""

    source_path = Path(image_path).expanduser().resolve()
    image = load_image(source_path)
    resized_image, metadata = resize_image(image, minimum_long_side=minimum_long_side)
    edge_map = generate_canny_edge_map(
        resized_image,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    edge_map_path, temp_dir = save_edge_map(edge_map, source_path=source_path, temp_root=temp_root)

    return PreprocessResult(
        source_path=source_path,
        edge_map_path=edge_map_path,
        temp_dir=temp_dir,
        metadata={
            **metadata,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "edge_map_width": int(edge_map.shape[1]),
            "edge_map_height": int(edge_map.shape[0]),
        },
    )
