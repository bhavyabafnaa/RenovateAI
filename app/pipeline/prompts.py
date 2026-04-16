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
    "photorealistic professionally designed interior concept for the same room, fully "
    "furnished and professionally staged {room_type}, designed in {style} style, elegant "
    "interior design, architectural photography"
)

LAYOUT_PROMPT = (
    "well-composed interior layout with balanced furniture placement, realistic spatial "
    "arrangement, furniture positioned naturally against walls or centered appropriately, "
    "interior design composition with focal points"
)

INTERIOR_LAYOUT_DIRECTIVE = (
    "interior staging layout with clearly visible furniture placement, primary furniture "
    "pieces positioned naturally in the room, balanced composition with focal furniture "
    "elements"
)

ROOM_PROMPTS = {
    "Living Room": (
        "designer sofa, elegant TV unit, media wall, coffee table, side tables, area rug, "
        "decorative wall panels, false ceiling design, recessed ceiling lights, cove "
        "lighting, pendant lights, curtains, stylish window treatment, decorative lamps, "
        "wall art, indoor plants, vases, display shelves"
    ),
    "Bedroom": (
        "beautiful bed, upholstered headboard, bedside tables, full-height wardrobe, study "
        "table, dressing table, vanity mirror, soft curtains, false ceiling design, warm "
        "layered lighting, area rug, elegant bedding, wall art, indoor plants, storage "
        "cabinets"
    ),
    "Kitchen": (
        "modular kitchen cabinets, upper and lower cabinets, premium countertop, backsplash "
        "design, island counter or breakfast counter if space allows, bar stools, organized "
        "utensils, countertop appliances, pendant lighting, under-cabinet lighting, dining "
        "table if layout supports it"
    ),
}

STYLE_PROMPTS = {
    "Modern Luxury": (
        "marble surfaces, premium wood panels, gold accents, warm ambient lighting, elegant "
        "textures, high-end luxury interior"
    ),
    "Scandinavian": (
        "light oak wood, minimal furniture, white walls, soft daylight, nordic interior "
        "design, cozy textures"
    ),
    "Japandi": (
        "warm wood tones, beige palette, japanese scandinavian fusion, minimalist decor, "
        "natural materials"
    ),
    "Minimal Contemporary": (
        "clean modern lines, neutral palette, sleek furniture, minimalist design, soft "
        "indirect lighting"
    ),
}

GLOBAL_SUFFIX = (
    "beautifully furnished, rich decor styling, layered lighting, interior design magazine "
    "photography, premium interior staging"
)

WIDE_ANGLE_INTERIOR_PHOTOGRAPHY = "wide-angle interior photography"

FURNITURE_PRIORITY_SUFFIX = (
    "fully furnished interior with prominent furniture pieces clearly visible, sofa, "
    "tables, cabinetry, decor elements, layered lighting, interior magazine style staging"
)

# Backward-compatible alias for existing imports.
GLOBAL_STAGING_SUFFIX = GLOBAL_SUFFIX

NEGATIVE_PROMPT = (
    "empty room, unfurnished room, minimal furniture, blank interior, bare interior, bare "
    "space, unfinished room, distorted furniture, floating objects, warped walls, "
    "unrealistic geometry, blurry, low quality"
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
        f"{INTERIOR_LAYOUT_DIRECTIVE}, "
        f"{ROOM_PROMPTS[canonical_room_type]}, "
        f"{STYLE_PROMPTS[canonical_style]}, "
        f"{FURNITURE_PRIORITY_SUFFIX}, "
        f"{WIDE_ANGLE_INTERIOR_PHOTOGRAPHY}"
    )
