#!/bin/bash
set -e

# ===============================
# Google Drive file id の固定設定
# ===============================
MODEL_DRIVE_ID="1iLqMGumnPQAPs6iyZU1zie_540ywjXWq"
MODEL_PATH="./best.pt"

echo "==== Starting AI Inference Server ===="

# ===============================
# 仮想環境があれば有効化、なければ作成して有効化
# ===============================
if [ -d ".venv" ]; then
  echo "Activating existing virtualenv .venv"
  . .venv/bin/activate
else
  echo "Creating virtualenv .venv and activating"
  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel
  # requirements は Build 時に入れている想定だが、念のためここで入れる（軽め）
  pip install -r requirements.txt || true
fi

# ===============================
# gdown インストール（venv 内に入れる）
# ===============================
pip install gdown --no-cache-dir

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
# サーバ起動（venv の python を使って実行）
# ===============================
echo "Launching Flask server with $(which python) ..."
exec python server.py
