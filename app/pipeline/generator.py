"""Diffusers-based ControlNet generator for renovation previews."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import logging
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from PIL import Image, ImageStat

from .prompts import NEGATIVE_PROMPT, build_prompt, get_style_definition

if TYPE_CHECKING:
    import torch
    from diffusers import StableDiffusionXLControlNetImg2ImgPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = Path("data") / "outputs"
logger = logging.getLogger(__name__)


def _normalize_runtime_device(device: str | None) -> str:
    """Normalize a runtime device setting into a supported torch device family."""

    normalized = (device or "cuda").strip().lower()
    if normalized in {"cuda", "gpu"}:
        return "cuda"
    if normalized == "cpu":
        return "cpu"
    raise ValueError("Runtime device must be 'cuda' or 'cpu'.")


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean flag from the environment."""

    value = os.getenv(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value such as 1/0 or true/false.")


def _default_model_cache_dir() -> Path | None:
    """Prefer the large RunPod workspace mount for Hugging Face model caches."""

    workspace = Path("/workspace")
    if not workspace.is_dir():
        return None

    cache_dir = workspace / ".cache" / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir.resolve()


GENERATION_DEFAULTS = {
    "num_inference_steps": 30,
    "guidance_scale": 5.0,
    "controlnet_conditioning_scale": 0.9,
    "strength": 0.65,
    "guess_mode": False,
    "canny_low_threshold": 100,
    "canny_high_threshold": 200,
    "target_long_side": 1024,
}


@dataclass(frozen=True)
class ModelConfig:
    """Model identifiers and load-time options for the Diffusers pipeline."""

    base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"
    controlnet_model_id: str = "diffusers/controlnet-canny-sdxl-1.0"
    vae_model_id: str | None = "madebyollin/sdxl-vae-fp16-fix"
    variant: str | None = "fp16"
    use_safetensors: bool = True
    cache_dir: Path | None = field(default_factory=_default_model_cache_dir)


@dataclass(frozen=True)
class InferenceConfig:
    """Runtime parameters for a single image generation."""

    num_inference_steps: int = GENERATION_DEFAULTS["num_inference_steps"]
    guidance_scale: float = GENERATION_DEFAULTS["guidance_scale"]
    controlnet_conditioning_scale: float = GENERATION_DEFAULTS["controlnet_conditioning_scale"]
    num_images_per_prompt: int = 1
    seed: int | None = None
    strength: float = GENERATION_DEFAULTS["strength"]
    guess_mode: bool = GENERATION_DEFAULTS["guess_mode"]
    canny_low_threshold: int = GENERATION_DEFAULTS["canny_low_threshold"]
    canny_high_threshold: int = GENERATION_DEFAULTS["canny_high_threshold"]
    target_long_side: int = GENERATION_DEFAULTS["target_long_side"]
    size_multiple: int = 64
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution settings for the underlying model pipeline."""

    device: str = field(default_factory=lambda: _normalize_runtime_device(os.getenv("RENOVATEAI_DEVICE")))
    require_cuda: bool = field(default_factory=lambda: _env_bool("RENOVATEAI_REQUIRE_CUDA", False))


@dataclass(frozen=True)
class GenerationResult:
    """Structured result from a Diffusers generation run."""

    input_image_path: Path
    output_image_path: Path
    edge_map_path: Path
    prompt: str
    negative_prompt: str
    style: str
    width: int
    height: int
    seed: int | None
    model_config: dict[str, Any]
    inference_config: dict[str, Any]


def _resolve_output_dir(output_dir: str | Path) -> Path:
    """Resolve the configured output directory relative to the project root."""

    path = Path(output_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _normalize_dimension(value: int, multiple: int) -> int:
    """Round a dimension to the nearest valid multiple for SDXL."""

    rounded = int(round(value / multiple) * multiple)
    return max(multiple, rounded)


def _compute_target_size(
    width: int,
    height: int,
    target_long_side: int,
    size_multiple: int,
) -> tuple[int, int]:
    """Resize while preserving aspect ratio for SDXL-friendly generation dimensions."""

    if target_long_side <= 0:
        raise ValueError("target_long_side must be greater than zero.")
    if size_multiple <= 0:
        raise ValueError("size_multiple must be greater than zero.")

    longest_side = max(width, height)
    scale = target_long_side / float(longest_side)

    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))

    target_width = _normalize_dimension(scaled_width, size_multiple)
    target_height = _normalize_dimension(scaled_height, size_multiple)

    return target_width, target_height


def _load_rgb_image(image_path: str | Path) -> Image.Image:
    """Load an image from disk and convert it to RGB."""

    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    with Image.open(path) as image:
        return image.convert("RGB")


def load_generation_images(
    input_image_path: str | Path,
    edge_map_path: str | Path,
    inference_config: InferenceConfig,
) -> tuple[Image.Image, Image.Image, tuple[int, int]]:
    """Load the source image and edge map and align them to one generation size."""

    source_image = _load_rgb_image(input_image_path)
    conditioning_image = _load_rgb_image(edge_map_path)

    target_width, target_height = _compute_target_size(
        width=conditioning_image.width,
        height=conditioning_image.height,
        target_long_side=inference_config.target_long_side,
        size_multiple=inference_config.size_multiple,
    )
    target_size = (target_width, target_height)

    if source_image.size != target_size:
        source_image = source_image.resize(target_size, Image.Resampling.LANCZOS)
    if conditioning_image.size != target_size:
        conditioning_image = conditioning_image.resize(target_size, Image.Resampling.LANCZOS)

    return source_image, conditioning_image, target_size


def save_output_image(image: Image.Image, output_dir: str | Path, style: str, edge_map_path: str | Path) -> Path:
    """Save a generated image to ``data/outputs`` or another configured directory."""

    directory = _resolve_output_dir(output_dir)
    edge_name = Path(edge_map_path).stem.replace("_edges", "")
    style_slug = style.strip().lower().replace(" ", "-")
    output_path = directory / f"{edge_name}_{style_slug}_{uuid4().hex[:8]}.png"
    image.save(output_path, format="PNG")
    return output_path


def _summarize_output_image_pixels(image: Image.Image) -> dict[str, list[float] | list[int]]:
    """Return min, max, and mean pixel stats for debug logging and validation."""

    rgb_image = image.convert("RGB")
    stat = ImageStat.Stat(rgb_image)
    return {
        "min": [int(low) for low, _ in stat.extrema],
        "max": [int(high) for _, high in stat.extrema],
        "mean": [round(float(channel_mean), 2) for channel_mean in stat.mean],
    }


def _is_blank_output_image(image: Image.Image) -> bool:
    """Detect fully black or near-black low-variance outputs before saving them."""

    pixel_stats = _summarize_output_image_pixels(image)
    mean_brightness = sum(pixel_stats["mean"]) / 3.0
    max_channel_range = max(
        high - low for low, high in zip(pixel_stats["min"], pixel_stats["max"])
    )

    return mean_brightness < 3.0 and max_channel_range < 6.0


def get_runtime_status(runtime_config: RuntimeConfig | None = None) -> dict[str, Any]:
    """Return torch/CUDA runtime details for API health checks and diagnostics."""

    config = runtime_config or RuntimeConfig()
    configured_device = _normalize_runtime_device(config.device)

    try:
        import torch
    except ImportError as exc:
        return {
            "configured_device": configured_device,
            "require_cuda": config.require_cuda,
            "torch_available": False,
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "cuda_device_count": 0,
            "cuda_device_name": None,
            "selected_device": None,
            "error": str(exc),
        }

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_device_count > 0 else None
    selected_device = configured_device
    if configured_device == "cuda" and not cuda_available:
        selected_device = None if config.require_cuda else "cpu"

    return {
        "configured_device": configured_device,
        "require_cuda": config.require_cuda,
        "torch_available": True,
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
        "selected_device": selected_device,
        "error": None,
    }


class ControlNetSDXLGenerator:
    """Lazy-loading wrapper around Stable Diffusion XL ControlNet img2img generation."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        inference_config: InferenceConfig | None = None,
        runtime_config: RuntimeConfig | None = None,
    ) -> None:
        self.model_config = model_config or ModelConfig()
        self.inference_config = inference_config or InferenceConfig()
        self.runtime_config = runtime_config or RuntimeConfig()
        self._pipeline: StableDiffusionXLControlNetImg2ImgPipeline | None = None

    def _import_runtime_dependencies(self) -> tuple[Any, Any, Any, Any]:
        """Import torch and Diffusers components only when generation is requested."""

        try:
            import torch
            from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
        except ImportError as exc:
            raise ImportError(
                "Diffusers generation dependencies are missing. Install torch, diffusers, "
                "transformers, and safetensors before using app.pipeline.generator."
            ) from exc

        return torch, ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL

    def _resolve_device(self, torch_module: Any) -> str:
        """Resolve the runtime device, falling back to CPU when CUDA is unavailable."""

        device = _normalize_runtime_device(self.runtime_config.device)
        if device == "cuda" and torch_module.cuda.is_available():
            return device
        if device == "cuda" and not torch_module.cuda.is_available() and not self.runtime_config.require_cuda:
            return "cpu"
        if self.runtime_config.require_cuda:
            raise RuntimeError(
                "CUDA GPU required for the default SDXL + ControlNet pipeline. "
                "CPU execution is typically too slow and memory-heavy for practical use."
            )
        return "cpu"

    def load_pipeline(self) -> StableDiffusionXLControlNetImg2ImgPipeline:
        """Load the ControlNet + SDXL img2img pipeline the first time it is needed."""

        if self._pipeline is not None:
            return self._pipeline

        torch, ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL = self._import_runtime_dependencies()
        device = self._resolve_device(torch)
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        variant = self.model_config.variant if device == "cuda" else None
        cache_dir = str(self.model_config.cache_dir) if self.model_config.cache_dir is not None else None

        common_load_kwargs = {
            "torch_dtype": torch_dtype,
            "variant": variant,
            "use_safetensors": self.model_config.use_safetensors,
        }
        if cache_dir is not None:
            common_load_kwargs["cache_dir"] = cache_dir

        vae = None
        if self.model_config.vae_model_id:
            vae_load_kwargs = {
                "torch_dtype": torch_dtype,
                "use_safetensors": self.model_config.use_safetensors,
            }
            if cache_dir is not None:
                vae_load_kwargs["cache_dir"] = cache_dir
            vae = AutoencoderKL.from_pretrained(
                self.model_config.vae_model_id,
                **vae_load_kwargs,
            )

        # CUDA remains the intended path for practical inference speed.
        # When CUDA is unavailable, the fallback uses float32 weights and `pipe.to("cpu")`.
        controlnet = ControlNetModel.from_pretrained(
            self.model_config.controlnet_model_id,
            **common_load_kwargs,
        )
        pipeline_load_kwargs = {
            "controlnet": controlnet,
            **common_load_kwargs,
        }
        if vae is not None:
            pipeline_load_kwargs["vae"] = vae
        pipeline = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            self.model_config.base_model_id,
            **pipeline_load_kwargs,
        )
        logger.info("Selected pipeline class: %s", pipeline.__class__.__name__)
        logger.info("VAE model ID: %s", self.model_config.vae_model_id or "pipeline default")
        pipeline.enable_attention_slicing()
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()
        pipeline = pipeline.to("cuda" if device == "cuda" else "cpu")
        pipeline.set_progress_bar_config(disable=True)

        self._pipeline = pipeline
        return pipeline

    def _build_torch_generator(self, torch_module: Any, device: str) -> Any | None:
        """Create a seeded torch generator when a seed is configured."""

        if self.inference_config.seed is None:
            return None
        return torch_module.Generator(device=device).manual_seed(self.inference_config.seed)

    def generate(
        self,
        input_image_path: str | Path,
        edge_map_path: str | Path,
        style: str,
        negative_prompt: str = NEGATIVE_PROMPT,
    ) -> GenerationResult:
        """Generate a single renovation image from the source room image and edge map."""

        if self.inference_config.num_images_per_prompt != 1:
            raise ValueError("This generator currently supports exactly one output image per request.")
        if not 0 < self.inference_config.strength <= 1:
            raise ValueError("strength must be between 0 and 1.")

        pipeline = self.load_pipeline()
        torch, _, _, _ = self._import_runtime_dependencies()
        device = self._resolve_device(torch)

        style_definition = get_style_definition(style)
        prompt = build_prompt(style_definition.key)
        source_image, conditioning_image, (target_width, target_height) = load_generation_images(
            input_image_path=input_image_path,
            edge_map_path=edge_map_path,
            inference_config=self.inference_config,
        )
        logger.info("Init image size/mode: %s / %s", source_image.size, source_image.mode)
        logger.info("Control image size/mode: %s / %s", conditioning_image.size, conditioning_image.mode)
        torch_generator = self._build_torch_generator(torch, device)
        autocast_context = torch.autocast(device) if device == "cuda" else nullcontext()

        # CUDA is the fast path. The CPU fallback works as a backup but will be much slower.
        with torch.inference_mode():
            with autocast_context:
                output = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=source_image,
                    control_image=conditioning_image,
                    width=target_width,
                    height=target_height,
                    strength=self.inference_config.strength,
                    num_inference_steps=self.inference_config.num_inference_steps,
                    guidance_scale=self.inference_config.guidance_scale,
                    controlnet_conditioning_scale=self.inference_config.controlnet_conditioning_scale,
                    guess_mode=self.inference_config.guess_mode,
                    num_images_per_prompt=1,
                    generator=torch_generator,
                )
        output_image = output.images[0]
        output_pixel_stats = _summarize_output_image_pixels(output_image)
        logger.info(
            "Output image pixel stats before save: min=%s max=%s",
            output_pixel_stats["min"],
            output_pixel_stats["max"],
        )
        if _is_blank_output_image(output_image):
            raise RuntimeError(
                "Generated image appears blank or nearly black. "
                "This usually indicates an SDXL runtime or VAE precision issue."
            )
        output_image_path = save_output_image(
            image=output_image,
            output_dir=self.inference_config.output_dir,
            style=style,
            edge_map_path=edge_map_path,
        )

        return GenerationResult(
            input_image_path=Path(input_image_path).expanduser().resolve(),
            output_image_path=output_image_path,
            edge_map_path=Path(edge_map_path).expanduser().resolve(),
            prompt=prompt,
            negative_prompt=negative_prompt,
            style=style_definition.key,
            width=target_width,
            height=target_height,
            seed=self.inference_config.seed,
            model_config=asdict(self.model_config),
            inference_config={
                **asdict(self.inference_config),
                "output_dir": str(_resolve_output_dir(self.inference_config.output_dir)),
            },
        )


def generate_image(
    input_image_path: str | Path,
    edge_map_path: str | Path,
    style: str,
    model_config: ModelConfig | None = None,
    inference_config: InferenceConfig | None = None,
    runtime_config: RuntimeConfig | None = None,
    negative_prompt: str = NEGATIVE_PROMPT,
) -> GenerationResult:
    """Convenience wrapper for one-off image generation."""

    generator = ControlNetSDXLGenerator(
        model_config=model_config,
        inference_config=inference_config,
        runtime_config=runtime_config,
    )
    return generator.generate(
        input_image_path=input_image_path,
        edge_map_path=edge_map_path,
        style=style,
        negative_prompt=negative_prompt,
    )
