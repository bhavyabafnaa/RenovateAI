"""Image normalization before generation."""

from PIL import Image, ImageOps

MAX_IMAGE_SIZE = (1024, 1024)


def prepare_image(image: Image.Image) -> Image.Image:
    prepared = ImageOps.exif_transpose(image).convert("RGB")
    prepared.thumbnail(MAX_IMAGE_SIZE)
    return prepared
