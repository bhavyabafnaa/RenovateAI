"""Prompt builders for interior concept generation."""

from __future__ import annotations

ROOM_TYPES = [
    "Living Room",
    "Bedroom",
    "Kitchen",
]

STYLES = [
    "Modern Luxury",
    "Scandinavian",
    "Japandi",
    "Minimal Contemporary",
]

BASE_PROMPT = (
    "photorealistic interior design concept for the same empty room, designed as a "
    "{room_type}, in {style} style, fully furnished, premium materials, elegant lighting, "
    "professional architectural interior photography"
)

ROOM_PROMPTS = {
    "Living Room": (
        "designer sofa, coffee table, TV wall unit, wall panels, area rug, curtains, "
        "decorative lighting"
    ),
    "Bedroom": (
        "queen size bed, bedside tables, wardrobe, soft lighting, curtains, wall decor"
    ),
    "Kitchen": (
        "modular kitchen cabinets, countertop, backsplash, storage cabinets, modern appliances"
    ),
}

STYLE_PROMPTS = {
    "Modern Luxury": (
        "marble surfaces, warm recessed lighting, elegant wall panels, premium materials, "
        "gold accents"
    ),
    "Scandinavian": (
        "light oak wood, white walls, minimalist furniture, natural daylight, Nordic design"
    ),
    "Japandi": (
        "warm wood tones, beige palette, Japanese Scandinavian fusion, minimalist decor"
    ),
    "Minimal Contemporary": (
        "clean modern lines, neutral palette, sleek cabinetry, hidden lighting"
    ),
}

NEGATIVE_PROMPT = (
    "distorted geometry, warped walls, bad perspective, floating furniture, duplicate objects, "
    "blurry, low resolution, cartoon style, unrealistic lighting"
)

# Backward-compatible alias for existing imports.
SHARED_NEGATIVE_PROMPT = NEGATIVE_PROMPT


def _normalize_choice(value: str, choices: list[str], field_name: str) -> str:
    normalized_value = value.strip().lower()
    for choice in choices:
        if choice.lower() == normalized_value:
            return choice

    supported = ", ".join(choices)
    raise ValueError(f"Unsupported {field_name} '{value}'. Choose from: {supported}.")


def normalize_room_type(room_type: str) -> str:
    """Return the canonical room type label for a supported room type."""

    return _normalize_choice(room_type, ROOM_TYPES, "room type")


def normalize_style(style: str) -> str:
    """Return the canonical style label for a supported style."""

    return _normalize_choice(style, STYLES, "style")


def list_room_types() -> list[str]:
    """Return the supported room types."""

    return list(ROOM_TYPES)


def list_styles() -> list[str]:
    """Return the supported interior concept styles."""

    return list(STYLES)


def build_prompt(room_type: str, style: str) -> str:
    """Build an interior concept prompt for the requested room type and style."""

    canonical_room_type = normalize_room_type(room_type)
    canonical_style = normalize_style(style)

    base_prompt = BASE_PROMPT.format(
        room_type=canonical_room_type,
        style=canonical_style,
    )
    return (
        f"{base_prompt}, "
        f"{ROOM_PROMPTS[canonical_room_type]}, "
        f"{STYLE_PROMPTS[canonical_style]}"
    )
