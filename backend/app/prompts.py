"""Compatibility prompt helpers for supported interior concept styles."""

from dataclasses import dataclass

from app.pipeline.prompts import STYLE_PROMPTS, STYLES, normalize_style


@dataclass(frozen=True)
class StylePrompt:
    key: str
    label: str
    prompt: str


def list_styles() -> list[StylePrompt]:
    return [
        StylePrompt(key=style, label=style, prompt=STYLE_PROMPTS[style])
        for style in STYLES
    ]


def get_style_prompt(style: str) -> StylePrompt:
    canonical_style = normalize_style(style)
    return StylePrompt(
        key=canonical_style,
        label=canonical_style,
        prompt=STYLE_PROMPTS[canonical_style],
    )
