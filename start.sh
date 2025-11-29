#!/bin/bash
set -e

# Render の Environment に MODEL_DRIVE_ID を設定してください（Google Drive file id）
# 例: MODEL_DRIVE_ID=1AbCDeFgHIjkLMnoPqRstuVwXyz

MODEL_PATH="./best.pt"

# インストール（軽量）: gdown がなければここで入れる
# （Build 時にも入れておくなら不要。安全のため毎回入れても OK）
python -m pip install --upgrade pip
python -m pip install gdown --no-cache-dir

if [ ! -f "$MODEL_PATH" ]; then
  if [ -n "$MODEL_DRIVE_ID" ]; then
    echo "Downloading best.pt from Google Drive (file id: $MODEL_DRIVE_ID) ..."
    # gdown は Google Drive の大きなファイルも扱える
    python -m gdown "https://drive.google.com/uc?id=${MODEL_DRIVE_ID}" -O "$MODEL_PATH"
    echo "Model downloaded to $MODEL_PATH"
  elif [ -n "$MODEL_URL" ]; then
    echo "Downloading best.pt from MODEL_URL ..."
    curl -L "$MODEL_URL" -o "$MODEL_PATH"
    echo "Model downloaded to $MODEL_PATH"
  else
    echo "Error: MODEL_DRIVE_ID or MODEL_URL must be set in environment, and best.pt not found locally."
    # 続行させるか停止するか選べます。ここでは続行してサーバを起動（推奨は停止して確認）
    # exit 1
  fi
else
  echo "Model already exists: $MODEL_PATH"
fi

# サーバ起動
exec python server.py
