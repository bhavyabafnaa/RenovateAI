"""Helpers for saving and exporting images."""

from base64 import b64encode
from io import BytesIO
from pathlib import Path
import re
import tempfile
from uuid import uuid4

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError


def load_image_from_bytes(data: bytes) -> Image.Image:
    """Load an image from raw bytes."""

    if not data:
        raise ValueError("The uploaded file is empty.")

    try:
        image = Image.open(BytesIO(data))
        image.load()
        return image
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc


def image_to_base64_png(image: Image.Image) -> str:
    """Encode a PIL image as base64 PNG."""

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return b64encode(buffer.getvalue()).decode("utf-8")


def _safe_filename(filename: str | None) -> str:
    """Create a filesystem-safe filename."""

    original_name = filename or "upload.png"
    cleaned_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original_name).name).strip("._")
    return cleaned_name or "upload.png"


async def save_upload_file(upload: UploadFile, root_dir: str | Path | None = None) -> Path:
    """Persist an uploaded file to a temporary folder and return the saved path."""

    data = await upload.read()
    if not data:
        raise ValueError("The uploaded file is empty.")

    root = Path(root_dir) if root_dir is not None else Path(tempfile.gettempdir()) / "renovateai-inputs"
    root.mkdir(parents=True, exist_ok=True)

    file_path = root / f"{uuid4().hex}_{_safe_filename(upload.filename)}"
    file_path.write_bytes(data)
    return file_path.resolve()
