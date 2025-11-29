#!/bin/bash
set -e

# Google Drive file id（もし環境変数で渡すなら優先的に使う）
: "${MODEL_DRIVE_ID:=1iLqMGumnPQAPs6iyZU1zie_540ywjXWq}"
MODEL_PATH="./best.pt"

echo "==== Start script ===="

# ▼ requirements.txt をインストール（これが最重要）
echo "Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir

# ▼ gdown を追加インストール
pip install gdown --no-cache-dir

# ▼ best.pt を Google Drive から取得
if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading best.pt from Google Drive (id: ${MODEL_DRIVE_ID}) ..."
  python -m gdown "https://drive.google.com/uc?id=${MODEL_DRIVE_ID}" -O "$MODEL_PATH"
  echo "Model saved to $MODEL_PATH"
else
  echo "Model already exists: $MODEL_PATH"
fi

# ▼ サーバ起動
echo "Launching Flask server..."
exec python server.py
