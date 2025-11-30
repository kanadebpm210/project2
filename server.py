import os
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
import numpy as np
from io import BytesIO
import traceback

# -----------------------------
# 環境変数 / ポート設定
# -----------------------------
PORT = int(os.environ.get("PORT", 10000))

BEST_MODEL = "best.onnx"      # バッテリー専用モデル（1クラス）
OTHER_MODEL = "other.onnx"    # YOLOv8n（80クラス）

app = Flask(__name__)

# -----------------------------
# モデル読み込み関数
# -----------------------------
def load_model(path):
    try:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        shape = session.get_inputs()[0].shape
        height, width = shape[2], shape[3]
        print(f"Loaded {path}: input={height}x{width}")
        return session, input_name, height, width
    except Exception:
        traceback.print_exc()
        return None, None, None, None

session_battery, input_battery, battery_h, battery_w = load_model(BEST_MODEL)
session_other, input_other, other_h, other_w = load_model(OTHER_MODEL)

# -----------------------------
# YOLO 前処理（RGB, resize, normalize, CHW）
# -----------------------------
def preprocess(img, h, w):
    img = img.convert("RGB")
    img = img.resize((w, h))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = np.expand_dims(arr, 0)
    return arr

# -----------------------------
# YOLO (other) の最大スコア計算
# -----------------------------
def get_other_score(output):
    preds = output[0]   # (N, 85)
    if preds.size == 0:
        return 0.0

    cls_scores = preds[:, 5:]  # (N,80)
    max_score = float(np.max(cls_scores))
    return max_score

# -----------------------------
# Batteryモデルのスコア（1値）
# -----------------------------
def get_battery_score(output):
    # 期待形状: [[ score ]]
    try:
        return float(output[0][0])
    except:
        return 0.0

# -----------------------------
# メイン /predict
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "image required"}), 400

        try:
            img = Image.open(BytesIO(request.files["image"].read()))
        except UnidentifiedImageError:
            return jsonify({"error": "Invalid image"}), 400

        # バッテリー推論
        inp_b = preprocess(img, battery_h, battery_w)
        out_b = session_battery.run(None, {input_battery: inp_b})
        score_battery = get_battery_score(out_b)

        # other推論
        inp_o = preprocess(img, other_h, other_w)
        out_o = session_other.run(None, {input_other: inp_o})
        score_other = get_other_score(out_o)

        # -------------------------
        # unknown を返さない
        # battery / other の優先度は完全に同じ
        # -------------------------
        if score_battery >= score_other:
            result = "battery"
        else:
            result = "other"

        return jsonify({
            "result": result,
            "battery_score": score_battery,
            "other_score": score_other
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------------
# 健康チェック
# -----------------------------
@app.route("/")
def index():
    return jsonify({"status": "running"})

# -----------------------------
# Flask起動
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
