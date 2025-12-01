#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import traceback
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
import numpy as np
from io import BytesIO
import requests

# -----------------------------
# 基本設定
# -----------------------------
PORT = int(os.environ.get("PORT", 10000))
MODEL_PATH = os.environ.get("MODEL_PATH", "other.onnx")   # デフォルト other.onnx

# Supabase（任意）
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_TABLE = os.environ.get("SUPABASE_TABLE", "ai_results")

app = Flask(__name__)
CORS(app)


# -----------------------------
# モデル読み込み（堅牢）
# -----------------------------
def load_model(path):
    try:
        print("DEBUG cwd:", os.getcwd())
        print("DEBUG files:", os.listdir("."))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        print("Loading ONNX model:", path)
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        shape = input_info.shape  # 例: [1,3,640,640]

        # shapeから高さ・幅を取り出す（動的な場合はフォールバック）
        try:
            _, _, h, w = shape
            if h is None or w is None:
                raise ValueError("dynamic shape")
            h = int(h); w = int(w)
        except Exception:
            h, w = 640, 640
            print("⚠ input shape dynamic or unknown, fallback to 640x640")

        print(f"✔ Loaded: {path} | input_name={input_name} | shape={shape} -> use ({h},{w})")
        return session, input_name, h, w

    except Exception:
        print("❌ Failed to load ONNX model:")
        traceback.print_exc()
        return None, None, None, None


session, input_name, model_h, model_w = load_model(MODEL_PATH)


# -----------------------------
# 前処理: Pillow -> (1,3,H,W)
# -----------------------------
def preprocess(img: Image.Image, h: int, w: int):
    if img is None:
        raise ValueError("img is None")
    img = img.convert("RGB")
    img = img.resize((w, h))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, 0)  # 1,3,H,W
    return arr


# -----------------------------
# COCO 80クラス
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
# 後処理（出力次元に応じて flatten/squeeze）
# - (N,85) -> そのまま
# - (1,N,85) や (B,N,85) -> (B*N,85)
# - ndim>=4 -> collapse to (-1, D)
# - 空や想定外は [] を返す
# -----------------------------
def postprocess(output) -> list:
    try:
        if output is None or len(output) == 0:
            return []

        preds = output[0]
        preds = np.array(preds)

        # no data
        if preds.size == 0:
            return []
        if preds.ndim == 1:
            # 1次元は想定外
            return []

        # collapse to (M, D)
        if preds.ndim == 2:
            # already (M, D)
            pass
        elif preds.ndim == 3:
            B, N, D = preds.shape
            preds = preds.reshape(B * N, D)
        else:
            # ndim >= 4
            D = preds.shape[-1]
            preds = preds.reshape(-1, D)

        # sanity check
        if preds.size == 0 or preds.shape[1] < 6:
            return []

        obj = preds[:, 4:5]   # (M,1)
        cls = preds[:, 5:]    # (M,C)
        if cls.size == 0:
            return []

        scores = obj * cls    # (M,C)
        if scores.size == 0:
            return []

        flat_idx = int(scores.argmax())
        box_idx, cls_idx = np.unravel_index(flat_idx, scores.shape)
        cls_idx = int(cls_idx)

        if cls_idx < 0 or cls_idx >= len(COCO_CLASSES):
            return []

        return [COCO_CLASSES[cls_idx]]

    except Exception:
        traceback.print_exc()
        return []


# -----------------------------
# Supabase に保存（任意）
# -----------------------------
def save_to_supabase(labels: list) -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        # 未設定ならスキップ
        return False
    try:
        url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{SUPABASE_TABLE}"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        payload = {"labels": labels}
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        print("Supabase:", res.status_code, res.text)
        return res.status_code in (200, 201)
    except Exception:
        traceback.print_exc()
        return False


# -----------------------------
# /predict エンドポイント
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
        except UnidentifiedImageError:
            return jsonify({"error": "invalid image"}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"failed to open image: {str(e)}"}), 400

        try:
            inp = preprocess(img, model_h, model_w)
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"preprocess failed: {str(e)}"}), 500

        try:
            out = session.run(None, {input_name: inp})
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": f"inference error: {str(e)}"}), 500

        labels = postprocess(out)
        saved = save_to_supabase(labels)

        return jsonify({"result": labels, "saved": saved})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# -----------------------------
# 健康チェック / デバッグ
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "running", "model_loaded": session is not None})


@app.route("/_ls", methods=["GET"])
def ls():
    # デバッグ用: ディレクトリの一覧（公開環境では無効化推奨）
    try:
        files = []
        for f in os.listdir("."):
            try:
                files.append({"name": f, "size": os.path.getsize(f)})
            except Exception:
                files.append({"name": f, "size": None})
        return jsonify({"cwd": os.getcwd(), "files": files})
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "failed to list files"}), 500


# -----------------------------
if __name__ == "__main__":
    print("🔥 Flask Inference Server Ready")
    app.run(host="0.0.0.0", port=PORT)
