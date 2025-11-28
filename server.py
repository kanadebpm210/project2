# server.py - Render 用 完全版
# battery → other の優先順位で推論
# Supabase に認識名を送る
# Git LFS の best.pt を読み込む

import os
import requests
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

# ============================================================
# Render 対応 : YOLO の設定ディレクトリ（書き込み可能な場所）
# ============================================================
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.makedirs("/tmp/Ultralytics", exist_ok=True)

# ============================================================
# 環境変数
# ============================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")  # Git LFS で入れた best.pt
OTHER_MODEL = os.environ.get("OTHER_MODEL", "yolov8n.pt")  # Render のメモリ対策
PORT = int(os.environ.get("PORT", 10000))

# Supabase 環境変数
SUPABASE_URL = os.environ.get("https://njlztbylmzysvfmtwweh.supabase.co")  # 例: https://xxxx.supabase.co
SUPABASE_API_KEY = os.environ.get("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5qbHp0YnlsbXp5c3ZmbXR3d2VoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTA5NjkwMSwiZXhwIjoyMDc2NjcyOTAxfQ.uUdg3jv-GXSZ9GpC8eULMhW-NxWjCL7VH7kxClaLvkM")  # service_role
SUPABASE_TABLE = os.environ.get("SUPABASE_TABLE", "ai_results")

# ============================================================
# Flask
# ============================================================
app = Flask(__name__)

# ============================================================
# Supabase 送信用関数
# ============================================================
def push_to_supabase(class_name: str):
    if not SUPABASE_URL or not SUPABASE_API_KEY:
        print("Supabase 未設定：送信スキップ")
        return False

    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    data = {"class": class_name}

    try:
        res = requests.post(url, json=data, headers=headers, timeout=5)
        res.raise_for_status()
        print("Supabase 送信成功:", data)
        return True
    except Exception as e:
        print("Supabase 送信エラー:", e)
        return False

# ============================================================
# モデル読み込み
# ============================================================
print("Loading battery model:", MODEL_PATH)
battery_model = YOLO(MODEL_PATH)

print("Loading other model:", OTHER_MODEL)
other_model = YOLO(OTHER_MODEL)

# ============================================================
# 画像読み込み
# ============================================================
def read_image(file):
    img = Image.open(BytesIO(file.read())).convert("RGB")
    return img

# ============================================================
# ルートチェック
# ============================================================
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "YOLO inference server running"}), 200

# ============================================================
# 推論API
# ============================================================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "image file required"}), 400

    img = read_image(request.files["image"])

    # ----------------------------------------
    # 1. battery model（best.pt）を優先
    # ----------------------------------------
    battery_res = battery_model.predict(img, conf=0.5, verbose=False)[0]
    if len(battery_res.boxes) > 0:
        push_to_supabase("battery")
        return jsonify({"result": "battery"}), 200

    # ----------------------------------------
    # 2. fallback : other model
    # ----------------------------------------
    other_res = other_model.predict(img, conf=0.3, verbose=False)[0]
    if len(other_res.boxes) > 0:
        cls_id = int(other_res.boxes.cls[0])
        class_name = other_model.names[cls_id]
        push_to_supabase(class_name)
        return jsonify({"result": class_name}), 200

    # ----------------------------------------
    # 3. unknown
    # ----------------------------------------
    push_to_supabase("unknown")
    return jsonify({"result": "unknown"}), 200

# ============================================================
# 実行
# ============================================================
if __name__ == "__main__":
    print(f"Starting server on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)
