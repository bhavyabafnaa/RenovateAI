"""Streamlit frontend for the RenovateAI PoC."""

from __future__ import annotations

import os
from pathlib import Path

import requests
import streamlit as st

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
BACKEND_REQUEST_TIMEOUT = int(os.getenv("BACKEND_REQUEST_TIMEOUT", "300"))
FALLBACK_STYLES = ["modern", "scandinavian", "industrial"]
STYLE_LABELS = {
    "scandinavian": "Scandinavian Minimalist",
    "modern": "Modern Luxury",
    "industrial": "Industrial Loft",
}


def _format_style_label(style: str) -> str:
    """Convert a style key into a user-facing label."""

    return STYLE_LABELS.get(style, style.replace("_", " ").title())


def fetch_available_styles(backend_url: str) -> tuple[list[str], str | None]:
    """Fetch the supported style list from the backend health endpoint."""

    try:
        response = requests.get(f"{backend_url}/health", timeout=5)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return FALLBACK_STYLES, f"Backend health check failed: {exc}"

    styles = payload.get("styles")
    if not isinstance(styles, list) or not styles:
        return FALLBACK_STYLES, "Backend did not return a valid style list. Using fallback styles."

    return [str(style) for style in styles], None


def request_generation(backend_url: str, uploaded_file, style: str) -> dict[str, object]:
    """Send the uploaded image and selected style to the backend generate endpoint."""

    response = requests.post(
        f"{backend_url}/generate",
        files={
            "image": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type or "image/png",
            )
        },
        data={"style": style},
        timeout=BACKEND_REQUEST_TIMEOUT,
    )

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if response.ok:
        return payload

    detail = payload.get("detail") if isinstance(payload, dict) else None
    message = detail or f"Backend returned status {response.status_code}."
    raise RuntimeError(str(message))


def resolve_result_image(payload: dict[str, object]) -> str:
    """Resolve the generated output image path returned by the backend."""

    candidate = payload.get("output_image_path")
    if isinstance(candidate, str) and Path(candidate).exists():
        return candidate
    raise RuntimeError("Backend did not return a generated output image path.")


def render_metadata(payload: dict[str, object]) -> None:
    """Render backend metadata in a compact, readable layout."""

    with st.expander("Generation Metadata", expanded=True):
        summary_column, paths_column = st.columns(2)

        with summary_column:
            st.markdown(f"**Style:** `{payload.get('style', '-')}`")
            st.markdown(f"**Mode:** `{payload.get('generation_mode', '-')}`")
            preprocessing = payload.get("preprocessing")
            if isinstance(preprocessing, dict):
                resized_width = preprocessing.get("resized_width", "-")
                resized_height = preprocessing.get("resized_height", "-")
                st.markdown(f"**Resized Image:** `{resized_width} x {resized_height}`")

        with paths_column:
            st.markdown(f"**Input Image:** `{payload.get('input_image_path', '-')}`")
            st.markdown(f"**Output Image:** `{payload.get('output_image_path', '-')}`")
            st.markdown(f"**Edge Map:** `{payload.get('edge_map_path', '-')}`")

        prompt = payload.get("prompt")
        if prompt:
            st.markdown("**Prompt**")
            st.write(prompt)

        negative_prompt = payload.get("negative_prompt")
        if negative_prompt:
            st.markdown("**Negative Prompt**")
            st.write(negative_prompt)

        preprocessing = payload.get("preprocessing")
        if isinstance(preprocessing, dict):
            st.markdown("**Preprocessing**")
            st.json(preprocessing)


def main() -> None:
    """Render the RenovateAI Streamlit interface."""

    st.set_page_config(page_title="RenovateAI", layout="wide")
    st.title("RenovateAI")
    st.caption("Upload a room image, choose a renovation style, and inspect the generation output.")

    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL)
    styles, style_warning = fetch_available_styles(backend_url)
    style_options = {_format_style_label(style): style for style in styles}

    if style_warning:
        st.warning(style_warning)
    else:
        st.success("Backend is reachable and style options are synced.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    selected_label = st.selectbox("Choose Style", list(style_options))

    if st.button("Generate", type="primary"):
        if uploaded_file is None:
            st.warning("Upload an image before generating.")
            return

        status = st.empty()
        status.info("Submitting image to the backend...")

        try:
            with st.spinner("Running renovation pipeline..."):
                payload = request_generation(backend_url, uploaded_file, style_options[selected_label])
        except RuntimeError as exc:
            status.error(f"Generation failed: {exc}")
            return
        except requests.RequestException as exc:
            status.error(f"Backend request failed: {exc}")
            return

        try:
            result_image = resolve_result_image(payload)
        except RuntimeError as exc:
            status.error(str(exc))
            return

        status.success("Generation completed.")

        before_column, after_column = st.columns(2)
        with before_column:
            st.subheader("Original")
            st.image(uploaded_file.getvalue(), width="stretch")

        with after_column:
            st.subheader("Generated")
            st.image(result_image, width="stretch")

        render_metadata(payload)


if __name__ == "__main__":
    main()
