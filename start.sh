#!/bin/bash
set -e

echo "==== Start script ===="

MODEL_ID="${GDRIVE_MODEL_ID}"
MODEL_PATH="./best.onnx"

# 依存関係をインストール
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir gdown

# best.onnx が無ければダウンロード
if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading best.onnx..."
  gdown --fuzzy "https://drive.google.com/uc?id=${MODEL_ID}" -O "$MODEL_PATH"
fi

echo "Launching Flask server..."
exec python server.py
