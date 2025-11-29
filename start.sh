#!/bin/bash
set -e

# ===============================
# Google Drive file id の固定設定
# ===============================
MODEL_DRIVE_ID="1iLqMGumnPQAPs6iyZU1zie_540ywjXWq"
MODEL_PATH="./best.pt"

echo "==== Starting AI Inference Server ===="

# ===============================
# gdown インストール
# ===============================
python -m pip install --upgrade pip
python -m pip install gdown --no-cache-dir

# ===============================
# モデルが無ければ Google Drive からDL
# ===============================
if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading best.pt from Google Drive (id: ${MODEL_DRIVE_ID}) ..."
  python -m gdown "https://drive.google.com/uc?id=${MODEL_DRIVE_ID}" -O "$MODEL_PATH"
  echo "Model saved to $MODEL_PATH"
else
  echo "Model already exists: $MODEL_PATH"
fi

# ===============================
# サーバ起動
# ===============================
echo "Launching Flask server..."
exec python server.py
