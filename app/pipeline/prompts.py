"""Prompt builders for realistic renovation generation."""

from __future__ import annotations

from dataclasses import dataclass

BASE_PROMPT = (
    "photorealistic interior renovation of the same room, maintain overall room geometry "
    "and perspective, preserve window and door positions, realistic architectural proportions, "
    "professional interior design photography, idealistic renovation, light oak cabinetry, "
    "pale wood flooring, matte white walls, soft natural daylight, minimal modern furniture, "
    "clean lines, warm neutral palette, airy Nordic apartment aesthetic, premium materials, "
    "high detail, subtle renovation changes to materials, finishes, lighting and cabinetry "
    "while keeping the same room structure"
)

NEGATIVE_PROMPT = (
    "distorted geometry, extra windows, extra doors, warped floor, warped ceiling, "
    "deformed room, bad perspective, floating furniture, duplicate objects, clutter, "
    "fisheye distortion, blurry, low resolution, oversaturated, cartoon, illustration, "
    "CGI render, unrealistic lighting"
)

# Backward-compatible alias for existing imports.
SHARED_NEGATIVE_PROMPT = NEGATIVE_PROMPT


@dataclass(frozen=True)
class StyleDefinition:
    """Prompt fragments for a renovation style."""

    key: str
    label: str
    prompt: str
    strength_default: float


STYLE_PRESETS = {
    "Scandinavian Minimalist": {
        "prompt": (
            "scandinavian minimalist interior renovation, light oak flooring, white matte walls, "
            "soft natural daylight, minimal modern furniture, airy apartment design, "
            "warm neutral tones, simple textures, clean lines"
        ),
        "strength_default": 0.65,
    },
    "Modern Luxury": {
        "prompt": (
            "modern luxury apartment renovation, elegant neutral palette, warm recessed lighting, "
            "premium stone finishes, refined cabinetry appearance, upscale contemporary styling, "
            "subtle marble accents, sleek detailing"
        ),
        "strength_default": 0.65,
    },
    "Industrial Loft": {
        "prompt": (
            "industrial loft interior renovation, exposed concrete texture accents, "
            "matte black metal details, dark wood flooring, warm ambient lighting, "
            "urban apartment aesthetic, restrained modern furniture, textured materials"
        ),
        "strength_default": 0.65,
    },
}

_STYLE_DEFINITIONS = {
    "scandinavian": StyleDefinition(
        key="scandinavian",
        label="Scandinavian Minimalist",
        prompt=STYLE_PRESETS["Scandinavian Minimalist"]["prompt"],
        strength_default=STYLE_PRESETS["Scandinavian Minimalist"]["strength_default"],
    ),
    "modern": StyleDefinition(
        key="modern",
        label="Modern Luxury",
        prompt=STYLE_PRESETS["Modern Luxury"]["prompt"],
        strength_default=STYLE_PRESETS["Modern Luxury"]["strength_default"],
    ),
    "industrial": StyleDefinition(
        key="industrial",
        label="Industrial Loft",
        prompt=STYLE_PRESETS["Industrial Loft"]["prompt"],
        strength_default=STYLE_PRESETS["Industrial Loft"]["strength_default"],
    ),
}

_STYLE_ALIASES = {
    "scandinavian": "scandinavian",
    "scandinavian minimalist": "scandinavian",
    "modern": "modern",
    "modern luxury": "modern",
    "industrial": "industrial",
    "industrial loft": "industrial",
}


def list_styles() -> list[str]:
    """Return the supported renovation styles."""

    return ["modern", "scandinavian", "industrial"]


def get_style_definition(style: str) -> StyleDefinition:
    """Return the style preset for a supported key or display label."""

    normalized_style = style.strip().lower()
    style_key = _STYLE_ALIASES.get(normalized_style, normalized_style)

    try:
        return _STYLE_DEFINITIONS[style_key]
    except KeyError as exc:
        supported = ", ".join(sorted(_STYLE_DEFINITIONS))
        raise ValueError(f"Unsupported style '{style}'. Choose from: {supported}.") from exc


def build_prompt(style: str) -> str:
    """Build a renovation prompt for the given style."""

    style_definition = get_style_definition(style)
    return f"{BASE_PROMPT}, {style_definition.prompt}"
