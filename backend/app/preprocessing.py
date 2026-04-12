"""Image normalization before generation."""

from PIL import Image, ImageOps

MINIMUM_LONG_SIDE = 1024


def prepare_image(image: Image.Image) -> Image.Image:
    prepared = ImageOps.exif_transpose(image).convert("RGB")
    long_side = max(prepared.size)
    if long_side < MINIMUM_LONG_SIDE:
        scale = MINIMUM_LONG_SIDE / float(long_side)
        target_size = (
            max(1, int(round(prepared.width * scale))),
            max(1, int(round(prepared.height * scale))),
        )
        return prepared.resize(target_size, Image.Resampling.LANCZOS)
    return prepared
