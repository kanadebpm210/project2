#!/bin/bash
set -e
pip install onnxruntime
echo "==== Start script ===="

MODEL_ID="${GDRIVE_MODEL_ID}"
MODEL_PATH="./best.onnx"

# Install gdown
pip install --no-cache-dir gdown

# Download best.onnx if not exists
if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading best.onnx from Google Drive..."
  gdown --fuzzy "https://drive.google.com/uc?id=${MODEL_ID}" -O "$MODEL_PATH"
  echo "Download complete: $MODEL_PATH"
else
  echo "best.onnx already exists."
fi

echo "Launching Flask server..."
exec python server.py
