"""Pipeline utilities for RenovateAI."""

from .generator import (
    ControlNetSDXLGenerator,
    GenerationResult,
    InferenceConfig,
    ModelConfig,
    RuntimeConfig,
    generate_image,
    get_runtime_status,
)
from .prompts import (
    ROOM_PROMPTS,
    ROOM_TYPES,
    SHARED_NEGATIVE_PROMPT,
    STYLES,
    STYLE_PROMPTS,
    build_prompt,
    list_room_types,
    list_styles,
)

__all__ = [
    "ControlNetSDXLGenerator",
    "GenerationResult",
    "InferenceConfig",
    "ModelConfig",
    "PreprocessResult",
    "RuntimeConfig",
    "ROOM_PROMPTS",
    "ROOM_TYPES",
    "SHARED_NEGATIVE_PROMPT",
    "STYLES",
    "STYLE_PROMPTS",
    "build_prompt",
    "generate_image",
    "get_runtime_status",
    "list_room_types",
    "list_styles",
    "preprocess_image",
]


def __getattr__(name: str):
    if name in {"PreprocessResult", "preprocess_image"}:
        from .preprocess import PreprocessResult, preprocess_image

        return {
            "PreprocessResult": PreprocessResult,
            "preprocess_image": preprocess_image,
        }[name]

    raise AttributeError(f"module 'app.pipeline' has no attribute {name!r}")
