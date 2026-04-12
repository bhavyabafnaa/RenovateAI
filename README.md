# RenovateAI PoC

Minimal full-stack proof of concept for AI interior concept previews from empty room images.

## Structure

```text
backend/
  app/
    main.py
    prompts.py
    preprocessing.py
    generation.py
    file_io.py
frontend/
  app.py
requirements.txt
```

## What it does

- FastAPI backend with `GET /health` and `POST /generate`
- Streamlit frontend with image upload, room type/style selection, generate button, and before/after display
- Image preprocessing, prompt construction, and ControlNet SDXL generation orchestration for interior concepts

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend:

   ```bash
   uvicorn backend.app.main:app --reload
   ```

4. In a second terminal, start the frontend:

   ```bash
   streamlit run frontend/app.py
   ```

5. Open the Streamlit URL shown in the terminal and upload an image.

## Run on Google Colab

If you do not have a local CUDA GPU, use the notebook at [renovateai_colab_backend.ipynb](c:/Users/_bhavyabafna/Desktop/RenovateAI/notebooks/renovateai_colab_backend.ipynb).

Basic flow:

1. Zip this `RenovateAI` folder locally.
2. Open the Colab notebook and switch the runtime to a GPU.
3. Upload the zip when prompted.
4. Enter a Hugging Face token after accepting the SDXL model license.
5. Copy the public backend URL printed by the notebook.
6. Run Streamlit locally with `BACKEND_URL` set to that URL.

## Run on RunPod

This repo includes a RunPod bootstrap script at `scripts/pod_start_full_stack.sh`.

Basic flow:

1. SSH into the pod.
2. `cd /workspace/RenovateAI`
3. Start the stack:

   ```bash
   bash scripts/pod_start_full_stack.sh
   ```

Notes:

- The script installs the non-Torch Python dependencies from `requirements.txt`.
- It then installs a CUDA 12.8-compatible PyTorch build so the default RunPod RTX 4090 image can use the GPU.
- Expose port `8501` in the RunPod pod settings to access the Streamlit frontend from your browser.
- The FastAPI backend stays on `127.0.0.1:8000` and is consumed by Streamlit on the same pod.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Notes

- The current `/generate` implementation requires the configured model-backed generation pipeline.
- No authentication, database, Docker, or background job system is included in this PoC.
