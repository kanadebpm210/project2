# server.py - Render 用（修正版）
import os
import requests
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

# YOLO 設定ディレクトリ（書き込み可能な場所）
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
os.makedirs("/tmp/Ultralytics", exist_ok=True)

# 環境変数（名前で取得）
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
OTHER_MODEL = os.environ.get("OTHER_MODEL", "yolov8n.pt")
PORT = int(os.environ.get("PORT", "10000"))
MODEL_URL = os.environ.get("MODEL_URL")  # 例: https://.../best.pt

# Supabase 環境変数（正しく取得）
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_API_KEY = os.environ.get("SUPABASE_API_KEY")
SUPABASE_TABLE = os.environ.get("SUPABASE_TABLE", "ai_results")

app = Flask(__name__)

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

# 起動時にモデルが無ければ MODEL_URL から取得
def ensure_model(path: str):
    if os.path.exists(path):
        print(f"Model found: {path}")
        return True
    if not MODEL_URL:
        print("MODEL_URL not set and model not present.")
        return False
    print(f"Downloading model from {MODEL_URL} to {path} ...")
    try:
        r = requests.get(MODEL_URL, stream=True, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Model download completed.")
        return True
    except Exception as e:
        print("Model download failed:", e)
        return False

# モデル準備
if not ensure_model(MODEL_PATH):
    print("Warning: model not ready. Server will still start but predictions will fail until model is present.")

print("Loading battery model:", MODEL_PATH)
battery_model = YOLO(MODEL_PATH)

print("Loading other model:", OTHER_MODEL)
other_model = YOLO(OTHER_MODEL)

def read_image(file):
    img = Image.open(BytesIO(file.read())).convert("RGB")
    return img

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "YOLO inference server running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "image file required"}), 400
    img = read_image(request.files["image"])
    try:
        battery_res = battery_model.predict(img, conf=0.5, verbose=False)[0]
    except Exception as e:
        print("battery model inference error:", e)
        battery_res = None
    if battery_res and len(battery_res.boxes) > 0:
        push_to_supabase("battery")
        return jsonify({"result": "battery"}), 200
    try:
        other_res = other_model.predict(img, conf=0.3, verbose=False)[0]
    except Exception as e:
        print("other model inference error:", e)
        other_res = None
    if other_res and len(other_res.boxes) > 0:
        cls_id = int(other_res.boxes.cls[0])
        class_name = other_model.names[cls_id]
        push_to_supabase(class_name)
        return jsonify({"result": class_name}), 200
    push_to_supabase("unknown")
    return jsonify({"result": "unknown"}), 200

if __name__ == "__main__":
    print(f"Starting server on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)
