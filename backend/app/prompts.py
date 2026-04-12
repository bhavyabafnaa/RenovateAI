"""Prompt helpers for style-specific generation instructions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StylePrompt:
    key: str
    label: str
    prompt: str


_STYLE_PROMPTS = {
    "modern": StylePrompt(
        key="modern",
        label="Modern Refresh",
        prompt="Bright modern renovation with clean lines, natural light, and uncluttered surfaces.",
    ),
    "cozy": StylePrompt(
        key="cozy",
        label="Cozy Warmth",
        prompt="Warm and inviting redesign with soft textures, warmer tones, and comfortable lighting.",
    ),
    "industrial": StylePrompt(
        key="industrial",
        label="Industrial Loft",
        prompt="Industrial makeover with cool tones, sharper contrast, and a loft-inspired finish.",
    ),
}


def list_styles() -> list[StylePrompt]:
    return list(_STYLE_PROMPTS.values())


def get_style_prompt(style: str) -> StylePrompt:
    try:
        return _STYLE_PROMPTS[style]
    except KeyError as exc:
        supported = ", ".join(sorted(_STYLE_PROMPTS))
        raise ValueError(f"Unsupported style '{style}'. Choose from: {supported}.") from exc
