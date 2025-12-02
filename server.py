#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import requests
import numpy as np

from ultralytics import YOLO  # ultralytics パッケージが必要

# -----------------------------
# 基本設定
# -----------------------------
PORT = int(os.environ.get("PORT", 10000))
# .onnx でも .pt でもOK（例: "yolov8n.onnx" や "yolov8n.pt"）
MODEL_PATH = os.environ.get("MODEL_PATH", "other.onnx")

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_TABLE = os.environ.get("SUPABASE_TABLE", "ai_results")

app = Flask(__name__)
CORS(app)

# -----------------------------
# YOLO モデル読み込み
# -----------------------------
yolo_model = None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    # Ultralytics YOLO にそのまま渡す (.onnx / .pt 両対応)
    yolo_model = YOLO(MODEL_PATH)
    print(f"✅ YOLO model loaded from {MODEL_PATH}")
except Exception:
    traceback.print_exc()
    yolo_model = None
    print("❌ Failed to load YOLO model")

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
# ヘルパー: テンソル/配列を numpy に
# -----------------------------
def to_numpy(x):
    """torch tensor or numpy array or list -> numpy.ndarray"""
    try:
        # pytorch tensor
        if hasattr(x, "cpu") and hasattr(x, "numpy"):
            return x.cpu().numpy()
        # numpy array
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.array(x)
    except Exception:
        return np.array(x)

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

    except Exception:
        traceback.print_exc()
        return False

# -----------------------------
# /predict
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # モデルがロードされていない
        if yolo_model is None:
            return jsonify({"error": "model not loaded"}), 500

        # 画像がない
        file = request.files.get("image")
        if file is None:
            return jsonify({"error": "image required"}), 400
        if file.filename == "":
            return jsonify({"error": "empty filename"}), 400

        # 画像として開く
        try:
            img = Image.open(file.stream).convert("RGB")
        except Exception:
            return jsonify({"error": "invalid image"}), 400

        # ★ Ultralytics YOLO にそのまま投げる（前処理・後処理は内部でやってくれる）
        results = yolo_model(img)   # returns Results object
        if len(results) == 0:
            return jsonify({"result": [], "saved": False})

        r0 = results[0]

        labels = []

        # r0.boxes may be a Boxes object; handle different shapes/types robustly
        try:
            # classes and confidences
            cls_arr = None
            conf_arr = None

            if hasattr(r0, "boxes") and r0.boxes is not None:
                # Boxes API: r0.boxes.cls, r0.boxes.conf (might be tensors or numpy)
                if hasattr(r0.boxes, "cls"):
                    cls_arr = to_numpy(r0.boxes.cls).ravel()
                if hasattr(r0.boxes, "conf"):
                    conf_arr = to_numpy(r0.boxes.conf).ravel()

                # fallback: some versions expose .data or .xyxy for box data
                if cls_arr is None and hasattr(r0.boxes, "data"):
                    data = to_numpy(r0.boxes.data)
                    # if last column is class id, try to read it
                    if data.shape[1] >= 6:
                        cls_arr = data[:, 5].astype(int)
                        conf_arr = data[:, 4].astype(float)

            # If we couldn't find arrays, try reading from r0.boxes (iterable)
            if cls_arr is None or conf_arr is None:
                # try iterating boxes (each box can be Box object with .cls, .conf)
                for b in getattr(r0, "boxes", []):
                    try:
                        c = to_numpy(getattr(b, "cls", None)).ravel()[0]
                        s = to_numpy(getattr(b, "conf", None)).ravel()[0]
                    except Exception:
                        # some box objects expose as lists
                        try:
                            c = float(b.cls) if hasattr(b, "cls") else None
                            s = float(b.conf) if hasattr(b, "conf") else None
                        except Exception:
                            continue
                    if c is not None and s is not None:
                        cls_arr = np.append(cls_arr, int(c)) if cls_arr is not None else np.array([int(c)])
                        conf_arr = np.append(conf_arr, float(s)) if conf_arr is not None else np.array([float(s)])

            # now build labels list
            if cls_arr is not None and conf_arr is not None and len(cls_arr) == len(conf_arr):
                for cid, score in zip(cls_arr.astype(int), conf_arr.astype(float)):
                    if 0 <= cid < len(COCO_CLASSES):
                        cls_name = COCO_CLASSES[cid]
                    else:
                        cls_name = str(cid)
                    labels.append({"class": cls_name, "score": float(score)})
        except Exception:
            traceback.print_exc()

        # Supabase へ保存（検出があった場合のみ）
        saved = False
        if labels:
            saved_any = False
            saved_details = []
            for l in labels:
                ok = save_to_supabase(l["class"])
                saved_any = saved_any or ok
                saved_details.append({"class": l["class"], "saved": ok})
            saved = saved_any
        else:
            saved_details = []

        return jsonify({"result": labels, "saved": saved, "saved_details": saved_details})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------------
# 健康チェック
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "running",
        "model_loaded": (yolo_model is not None),
        "model_path": MODEL_PATH
    })

# -----------------------------
# メイン
# -----------------------------
if __name__ == "__main__":
    print("🔥 Flask Inference Server Ready")
    app.run(host="0.0.0.0", port=PORT)
