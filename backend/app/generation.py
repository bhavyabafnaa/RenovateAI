"""Minimal image generation placeholder for the PoC."""

from PIL import Image, ImageEnhance, ImageFilter


def _blend_with_color(image: Image.Image, color: tuple[int, int, int], alpha: float) -> Image.Image:
    overlay = Image.new("RGB", image.size, color)
    return Image.blend(image, overlay, alpha)


def generate_design(image: Image.Image, style: str) -> Image.Image:
    result = image.copy()

    if style == "cozy":
        result = _blend_with_color(result, (222, 190, 150), 0.18)
        result = ImageEnhance.Color(result).enhance(1.15)
        result = ImageEnhance.Brightness(result).enhance(1.05)
        return result.filter(ImageFilter.SMOOTH)

    if style == "industrial":
        result = ImageEnhance.Color(result).enhance(0.65)
        result = ImageEnhance.Contrast(result).enhance(1.25)
        result = _blend_with_color(result, (120, 128, 140), 0.12)
        return result.filter(ImageFilter.DETAIL)

    result = ImageEnhance.Brightness(result).enhance(1.08)
    result = ImageEnhance.Contrast(result).enhance(1.1)
    result = _blend_with_color(result, (245, 245, 240), 0.08)
    return result.filter(ImageFilter.SHARPEN)
