"""FastAPI application for the RenovateAI PoC."""

import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.pipeline import get_runtime_status, list_styles

from .file_io import save_upload_file
from .generation_service import GenerationUnavailableError, generate_renovation, initialize_generator

logger = logging.getLogger(__name__)

app = FastAPI(title="RenovateAI API")


@app.on_event("startup")
def startup() -> None:
    """Warm the real generator once at process startup."""

    try:
        initialize_generator()
    except Exception:
        logger.exception("Real generator startup initialization failed.")


@app.get("/health")
def health() -> dict[str, object]:
    runtime = get_runtime_status()
    status = "ok"
    if runtime["require_cuda"] and not runtime["cuda_available"]:
        status = "gpu_unavailable"

    return {"status": status, "styles": list_styles(), "runtime": runtime}


@app.post("/generate")
async def generate(image: UploadFile = File(...), style: str = Form(...)) -> dict[str, object]:
    try:
        saved_image_path = await save_upload_file(image)
        result = generate_renovation(saved_image_path, style)
    except GenerationUnavailableError as exc:
        logger.warning("Generation request failed because the real generator is unavailable.")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "style": result.style,
        "generation_mode": result.generation_mode,
        "prompt": result.prompt,
        "negative_prompt": result.negative_prompt,
        "input_image_path": str(result.artifacts.input_image_path),
        "output_image_path": str(result.artifacts.output_image_path),
        "edge_map_path": str(result.artifacts.edge_map_path),
        "temp_dir": str(result.artifacts.temp_dir),
        "preprocessing": result.preprocessing,
    }
