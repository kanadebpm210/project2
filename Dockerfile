# Dockerfile (Render 用・安定版)
FROM python:3.10-slim

# 必要な system package を入れる
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg git git-lfs build-essential && \
    rm -rf /var/lib/apt/lists/*

# git-lfs を有効化
RUN git lfs install

# 作業ディレクトリ
WORKDIR /app

# 依存ファイルを先にコピーしてキャッシュを活かす
COPY requirements.txt /app/requirements.txt

# pip 更新
RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch (CPU版) を事前に入れる — 互換性を確実にするために明示的に入れる
# ※ 必要なら GPU 用の wheel に差し替えてください（環境による）
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# その後に他の python パッケージを入れる
RUN pip install --no-cache-dir -r /app/requirements.txt

# アプリをコピー
COPY . /app

# 書き込み用ディレクトリ（Ultralytics 設定用）
RUN mkdir -p /tmp/Ultralytics
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

# start.sh を実行可能にする
RUN chmod +x /app/start.sh

EXPOSE 10000

# コンテナ起動時に start.sh を実行（start.sh はモデルをダウンロードして server.py を起動します）
CMD ["./start.sh"]
