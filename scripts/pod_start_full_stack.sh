#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${FRONTEND_PORT:-8501}"
BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"
BACKEND_LOG="${BACKEND_LOG:-/tmp/renovateai-backend.log}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0+cu128}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0+cu128}"
REBUILD_VENV="${REBUILD_VENV:-0}"
RENOVATEAI_DEVICE="${RENOVATEAI_DEVICE:-cuda}"
RENOVATEAI_REQUIRE_CUDA="${RENOVATEAI_REQUIRE_CUDA:-1}"

cd "$PROJECT_ROOT"

if [[ ! -f "requirements.txt" ]]; then
  echo "requirements.txt not found in $PROJECT_ROOT" >&2
  exit 1
fi

if [[ ! -f "backend/app/main.py" ]]; then
  echo "backend/app/main.py not found in $PROJECT_ROOT" >&2
  exit 1
fi

if [[ ! -f "streamlit_app.py" ]]; then
  echo "streamlit_app.py not found in $PROJECT_ROOT" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found on PATH" >&2
  exit 1
fi

if [[ "$REBUILD_VENV" == "1" && -d "$VENV_DIR" ]]; then
  rm -rf "$VENV_DIR"
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

base_requirements="$(mktemp)"
trap 'rm -f "$base_requirements"' EXIT
grep -Ev '^(torch|torchvision)([<>=!~].*)?$' requirements.txt >"$base_requirements"
python -m pip install -r "$base_requirements"

if ! python - <<'PY'
import os
import sys

try:
    import torch
    import torchvision
except Exception:
    sys.exit(1)

expected_torch = os.environ["TORCH_VERSION"]
expected_torchvision = os.environ["TORCHVISION_VERSION"]
ok = (
    torch.__version__ == expected_torch
    and torchvision.__version__ == expected_torchvision
    and torch.cuda.is_available()
)
sys.exit(0 if ok else 1)
PY
then
  python -m pip install --upgrade --force-reinstall \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "$TORCH_INDEX_URL"
fi

python - <<'PY'
import sys
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.stderr.write(
        "CUDA is not available for torch on this pod. "
        "Check the pod GPU/driver image or adjust TORCH_INDEX_URL and version pins.\n"
    )
    sys.exit(1)
print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

if command -v lsof >/dev/null 2>&1; then
  backend_pids="$(lsof -ti "tcp:${BACKEND_PORT}" || true)"
  frontend_pids="$(lsof -ti "tcp:${FRONTEND_PORT}" || true)"

  if [[ -n "$backend_pids" ]]; then
    kill $backend_pids >/dev/null 2>&1 || true
  fi
  if [[ -n "$frontend_pids" ]]; then
    kill $frontend_pids >/dev/null 2>&1 || true
  fi
fi

env RENOVATEAI_DEVICE="$RENOVATEAI_DEVICE" \
  RENOVATEAI_REQUIRE_CUDA="$RENOVATEAI_REQUIRE_CUDA" \
  PYTHONPATH="$PROJECT_ROOT" \
  python -m uvicorn backend.app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" >"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

cleanup() {
  if kill -0 "$BACKEND_PID" >/dev/null 2>&1; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

for _ in $(seq 1 60); do
  if curl -fsS "${BACKEND_URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! curl -fsS "${BACKEND_URL}/health" >/dev/null 2>&1; then
  echo "Backend did not become healthy at ${BACKEND_URL}/health" >&2
  echo "Backend log:" >&2
  cat "$BACKEND_LOG" >&2
  exit 1
fi

echo "Backend is healthy at ${BACKEND_URL}"
echo "Frontend will listen on ${FRONTEND_HOST}:${FRONTEND_PORT}"

exec env BACKEND_URL="$BACKEND_URL" \
  RENOVATEAI_DEVICE="$RENOVATEAI_DEVICE" \
  RENOVATEAI_REQUIRE_CUDA="$RENOVATEAI_REQUIRE_CUDA" \
  PYTHONPATH="$PROJECT_ROOT" \
  python -m streamlit run streamlit_app.py --server.address "$FRONTEND_HOST" --server.port "$FRONTEND_PORT"
