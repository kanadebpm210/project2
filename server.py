#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import traceback
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import requests
from datetime import datetime

# -----------------------------
# 基本設定
# -----------------------------
PORT = int(os.environ.get("PORT", 10000))
MODEL_PATH = os.environ.get("MODEL_PATH", "other.onnx")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_TABLE = os.environ.get("SUPABASE_TABLE", "ai_results")

app = Flask(__name__)
CORS(app)

# -----------------------------
# モデル読み込み
# -----------------------------
def load_model(path):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_info = session.get_inputs()[0]
        shape = input_info.shape
        try:
            _, _, h, w = shape
            if h is None or w is None:
                raise ValueError
            h, w = int(h), int(w)
        except:
            h, w = 640, 640
        return session, input_info.name, h, w
    except Exception:
        traceback.print_exc()
        return None, None, None, None

session, input_name, model_h, model_w = load_model(MODEL_PATH)

# -----------------------------
# 前処理
# -----------------------------
def preprocess(img, h, w):
    img = img.convert("RGB").resize((w, h))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, 0)

# -----------------------------
# COCOクラス
# -----------------------------
COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard',
    'cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase',
    'scissors','teddy bear','hair drier','toothbrush'
]

# -----------------------------
# 後処理
# -----------------------------
def postprocess(output):
    try:
        preds = np.array(output[0])
        if preds.size == 0 or preds.ndim < 2:
            return []
        if preds.ndim == 3:
            B, N, D = preds.shape
            preds = preds.reshape(B * N, D)
        elif preds.ndim >= 4:
            D = preds.shape[-1]
            preds = preds.reshape(-1, D)
        if preds.shape[1] < 6:
            return []
        obj = preds[:, 4:5]
        cls = preds[:, 5:]
        scores = obj * cls
        flat_idx = int(scores.argmax())
        multi_idx = np.unravel_index(flat_idx, scores.shape)
        cls_idx = multi_idx[-1]
        if cls_idx < 0 or cls_idx >= len(COCO_CLASSES):
            return []
        return [COCO_CLASSES[int(cls_idx)]]
    except:
        traceback.print_exc()
        return []

# -----------------------------
# Supabase 保存
# -----------------------------
def save_to_supabase(class_name: str) -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("⚠ Supabase URL/KEY not set. Skipping save.")
        return False
    try:
        url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{SUPABASE_TABLE}"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        payload = {
            "class": class_name,
            "created_at": datetime.utcnow().isoformat()  # UTC 時刻
        }
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        print("Supabase response:", res.status_code, res.text)
        return res.status_code in (200, 201)
    except:
        traceback.print_exc()
        return False

# -----------------------------
# /predict
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if session is None:
            return jsonify({"error": "model not loaded"}), 500
        if "image" not in request.files:
            return jsonify({"error": "image required"}), 400
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "empty filename"}), 400

        try:
            img = Image.open(file.stream)
        except:
            return jsonify({"error": "invalid image"}), 400

        inp = preprocess(img, model_h, model_w)
        out = session.run(None, {input_name: inp})
        labels = postprocess(out)
        class_name = labels[0] if labels else None
        saved = save_to_supabase(class_name) if class_name else False

        return jsonify({"result": labels, "saved": saved})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------------
# 健康チェック
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "running", "model_loaded": session is not None})

# -----------------------------
if __name__ == "__main__":
    print("🔥 Flask Inference Server Ready")
    app.run(host="0.0.0.0", port=PORT)
