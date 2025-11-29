FROM python:3.10-slim

# 基本ツールと git-lfs を入れる
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git-lfs build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN git lfs install

WORKDIR /app
COPY . /app

# pip とホイール更新
RUN python -m pip install --upgrade pip setuptools wheel

# 例: CPU 用 torch wheel 指定（必要に応じて調整）
# pipが自動で適切なtorchを選べるならrequirements.txtのままでも可。
RUN pip install -r requirements.txt

# gunicorn で起動（複数Workerはメモリに注意）
EXPOSE 10000
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:10000", "server:app"]
