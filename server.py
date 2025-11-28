# server.py (簡易版)
import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import requests

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["YOLO_CONFIG_DIR"] = os.environ.get("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
MODEL_URL  = os.environ.get("MODEL_URL")        # あればダウンロード
OTHER_MODEL = os.environ.get("OTHER_MODEL", "yolov8n.pt")

def download_if_needed():
    if os.path.exists(MODEL_PATH):
        return True
    if MODEL_URL:
        r = requests.get(MODEL_URL, stream=True, timeout=60)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return True
    return False

# 起動時にモデル準備
if not download_if_needed():
    print("MODEL not found. Set MODEL_PATH or MODEL_URL")
    # ここで続行するか例外で止めるか判断
try:
    battery_model = YOLO(MODEL_PATH)
except Exception as e:
    print("Battery model load failed:", e)
    battery_model = None
other_model = None
try:
    other_model = YOLO(OTHER_MODEL)
except Exception as e:
    print("Other model load failed:", e)

def load_image(f):
    return Image.open(BytesIO(f.read())).convert("RGB")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error":"image required"}), 400
    img = load_image(request.files["image"])
    # battery first
    if battery_model:
        res = battery_model.predict(img, conf=0.5, verbose=False)[0]
        if len(res.boxes):
            # supabase push optional (see below)
            return jsonify({"result":"battery"}), 200
    # other
    if other_model:
        res = other_model.predict(img, conf=0.3, verbose=False)[0]
        if len(res.boxes):
            cls = other_model.names[int(res.boxes.cls[0])]
            return jsonify({"result": cls}), 200
    return jsonify({"result":"unknown"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
