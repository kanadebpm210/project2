#!/bin/bash
set -e

# Google Drive file id（もし環境変数で渡すなら優先的に使う）
: "${MODEL_DRIVE_ID:=1iLqMGumnPQAPs6iyZU1zie_540ywjXWq}"
MODEL_PATH="./best.pt"

echo "==== Start script ===="

# gdown をインストール（コンテナ内の pip を使う）
python -m pip install --upgrade pip
python -m pip install gdown --no-cache-dir

# モデルダウンロード
if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading best.pt from Google Drive (id: ${MODEL_DRIVE_ID}) ..."
  python -m gdown "https://drive.google.com/uc?id=${MODEL_DRIVE_ID}" -O "$MODEL_PATH"
  echo "Model saved to $MODEL_PATH"
else
  echo "Model already exists: $MODEL_PATH"
fi

# サーバ起動
echo "Launching Flask server..."
exec python server.py
