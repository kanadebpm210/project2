#!/bin/bash
set -e

echo "==== Start script ===="

MODEL_ID="${GDRIVE_MODEL_ID}"
MODEL_PATH="./best.onnx"

# Install all dependencies from requirements.txt
echo "Installing dependencies..."
pip install --no-cache-dir -r requirements.txt

# Ensure gdown is installed for downloading model
pip install --no-cache-dir gdown

# Download best.onnx from Google Drive if it does not exist
if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading best.onnx from Google Drive..."
  gdown --fuzzy "https://drive.google.com/uc?id=${MODEL_ID}" -O "$MODEL_PATH"
  echo "Download complete: $MODEL_PATH"
else
  echo "best.onnx already exists."
fi

# Launch Flask server
echo "Launching Flask server..."
exec python server.py
