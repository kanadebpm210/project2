#!/bin/bash
set -euo pipefail

echo "==== start.sh ===="
echo "PWD: $(pwd)"
ls -lah

MODEL_PATH=${MODEL_PATH:-other.onnx}
MODEL_DRIVE_ID=${MODEL_DRIVE_ID:-}  # optional: Google Drive ID
PYTHON_CMD=${PYTHON_CMD:-python}

if [ ! -f "${MODEL_PATH}" ]; then
  if [ -n "${MODEL_DRIVE_ID}" ]; then
    echo "Model not found. Downloading from Drive id=${MODEL_DRIVE_ID} ..."
    ${PYTHON_CMD} - <<PY
import subprocess, sys
try:
    import gdown
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "gdown"])
    import gdown
gdown.download("https://drive.google.com/uc?id=${MODEL_DRIVE_ID}", "${MODEL_PATH}", quiet=False)
PY
  else
    echo "Model ${MODEL_PATH} not found and MODEL_DRIVE_ID not set. Server may fail to load model."
  fi
else
  echo "Model exists: ${MODEL_PATH} (size: $(stat -c%s "${MODEL_PATH}") bytes)"
fi

echo "Starting server..."
exec ${PYTHON_CMD} -u server.py
