"""Generation orchestration for the RenovateAI API."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from app.pipeline.generator import ControlNetSDXLGenerator
from app.pipeline.preprocess import preprocess_image
from app.pipeline.prompts import NEGATIVE_PROMPT, build_prompt, normalize_room_type, normalize_style

logger = logging.getLogger(__name__)
_GENERATOR: ControlNetSDXLGenerator | None = None


class GenerationUnavailableError(RuntimeError):
    """Raised when the real generator cannot be used in the current environment."""


@dataclass(frozen=True)
class GenerationArtifacts:
    """Paths produced during a generation request."""

    input_image_path: Path
    output_image_path: Path
    edge_map_path: Path
    temp_dir: Path


@dataclass(frozen=True)
class GenerationResult:
    """Serializable result for the generation endpoint."""

    room_type: str
    style: str
    prompt: str
    negative_prompt: str
    generation_mode: str
    artifacts: GenerationArtifacts
    preprocessing: dict[str, int | float | str]


def get_generator() -> ControlNetSDXLGenerator:
    """Return the shared generator instance for the running process."""

    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = ControlNetSDXLGenerator()
    return _GENERATOR


def initialize_generator() -> ControlNetSDXLGenerator:
    """Initialize the real generator once so requests can reuse the loaded pipeline."""

    generator = get_generator()
    logger.info(
        "Initializing real generator at startup with model IDs: base=%s controlnet=%s",
        generator.model_config.base_model_id,
        generator.model_config.controlnet_model_id,
    )
    generator.load_pipeline()
    logger.info("Real generator initialized successfully.")
    return generator


def generate_renovation(image_path: str | Path, room_type: str, style: str) -> GenerationResult:
    """Run preprocessing and attempt real generation for the requested interior concept."""

    input_image_path = Path(image_path).expanduser().resolve()
    canonical_room_type = normalize_room_type(room_type)
    canonical_style = normalize_style(style)

    logger.info(
        "Entering generation for room_type=%s style=%s input_image=%s",
        canonical_room_type,
        canonical_style,
        input_image_path,
    )
    prompt = build_prompt(canonical_room_type, canonical_style)
    preprocess_result = preprocess_image(input_image_path)

    generator = get_generator()
    logger.warning("Mock mode disabled. Real generation is required for /generate requests.")
    logger.info(
        "Chosen model IDs: base=%s controlnet=%s",
        generator.model_config.base_model_id,
        generator.model_config.controlnet_model_id,
    )

    try:
        generation_result = generator.generate(
            input_image_path=input_image_path,
            edge_map_path=preprocess_result.edge_map_path,
            room_type=canonical_room_type,
            style=canonical_style,
            negative_prompt=NEGATIVE_PROMPT,
        )
    except RuntimeError as exc:
        logger.exception("Real generation could not be completed for input_image=%s", input_image_path)
        raise GenerationUnavailableError(
            str(exc) or "Real generator failed during inference."
        ) from exc
    except (ImportError, OSError) as exc:
        logger.exception("Real generation could not be completed for input_image=%s", input_image_path)
        raise GenerationUnavailableError(
            "Real generator unavailable. Configure the Diffusers ControlNet SDXL pipeline before using /generate."
        ) from exc

    logger.info("Output file path: %s", generation_result.output_image_path)

    return GenerationResult(
        room_type=generation_result.room_type,
        style=generation_result.style,
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        generation_mode="real",
        artifacts=GenerationArtifacts(
            input_image_path=input_image_path,
            output_image_path=generation_result.output_image_path,
            edge_map_path=preprocess_result.edge_map_path,
            temp_dir=preprocess_result.edge_map_path.parent,
        ),
        preprocessing=preprocess_result.metadata,
    )
