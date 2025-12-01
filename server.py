#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import traceback
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import requests

# -----------------------------
# 基本設定
# -----------------------------
PORT = int(os.environ.get("PORT", 10000))
MODEL_PATH = os.environ.get("MODEL_PATH", "other.onnx")  # yolov8n.onnx
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
            h, w = int(h), int(w)
        except:
            h, w = 640, 640
        return session, input_info.name, h, w
    except Exception:
        traceback.print_exc()
        return None, None, None, None

session, input_name, model_h, model_w = load_model(MODEL_PATH)

# -----------------------------
# Letterbox 前処理
# -----------------------------
def letterbox_image(img, new_shape=(640, 640), color=(114, 114, 114)):
    img = np.array(img.convert("RGB"))
    h, w = img.shape[:2]
    new_w, new_h = new_shape
    scale = min(new_w / w, new_h / h)
    resize_w, resize_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (resize_w, resize_h))
    pad_w, pad_h = new_w - resize_w, new_h - resize_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_padded, scale, left, top

def preprocess(img, h, w):
    img_padded, scale, pad_x, pad_y = letterbox_image(img, (w, h))
    arr = img_padded.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, 0), scale, pad_x, pad_y

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
# 後処理（confidence 無視で全クラス返却）
# -----------------------------
def postprocess_debug(output):
    try:
        preds = np.array(output[0])
        print("DEBUG: output shape:", preds.shape)
        print("DEBUG: first 5 predictions:", preds[:5])
        if preds.size == 0:
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

        results = []
        for i in range(preds.shape[0]):
            cls_idx = int(np.argmax(cls[i]))
            conf = float(cls[i, cls_idx])
            results.append({
                "class": COCO_CLASSES[cls_idx],
                "score": conf
            })
        return results
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
        payload = {"class": class_name}  # 単一クラス名
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

        inp, scale, pad_x, pad_y = preprocess(img, model_h, model_w)
        out = session.run(None, {input_name: inp})
        labels = postprocess_debug(out)

        saved = False
        if labels:
            for l in labels:
                save_to_supabase(l["class"])
            saved = True

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
