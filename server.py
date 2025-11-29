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

# -----------------------------
# モデルファイル
# -----------------------------
BEST_MODEL = "best.onnx"   # batteryモデル
OTHER_MODEL = "other.onnx" # YOLOv8n ONNXモデル

# 推論閾値
BATTERY_THRESHOLD = 0.3
OTHER_THRESHOLD = 0.25

# Flaskアプリ作成
app = Flask(__name__)

# -----------------------------
# ONNXモデルロード
# -----------------------------
def load_model(path):
    try:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        height, width = input_shape[2], input_shape[3]
        return session, input_name, height, width
    except Exception:
        print(f"Failed to load {path}")
        traceback.print_exc()
        return None, None, None, None

session_battery, input_battery, battery_height, battery_width = load_model(BEST_MODEL)
session_other, input_other, other_height, other_width = load_model(OTHER_MODEL)

# -----------------------------
# other.onnx のラベルを取得
# -----------------------------
def get_class_names(session):
    try:
        meta = session.get_modelmeta().custom_metadata_map
        if "names" in meta:
            names = eval(meta["names"])  # 文字列リスト -> Python list
            return names
    except Exception:
        pass
    # fallback: 出力のチャンネル数から自動生成
    n_classes = session.get_outputs()[0].shape[1] - 5  # YOLOは最初5要素がbox info
    return [f"class{i}" for i in range(n_classes)]

other_names = get_class_names(session_other)

# -----------------------------
# 画像前処理
# -----------------------------
def preprocess(img, target_height, target_width):
    img = img.convert("RGB")
    img = img.resize((target_width, target_height))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, 0)
    return arr

# -----------------------------
# 出力後処理
# -----------------------------
def postprocess(output, names, threshold=0.5):
    try:
        preds = output[0][0]
        if len(preds) < 6:
            return "unknown", []

        cls_scores = preds[5:]
        max_score = float(np.max(cls_scores))
        cls_id = int(np.argmax(cls_scores))
        label = names[cls_id] if max_score >= threshold and cls_id < len(names) else "unknown"
        return label, cls_scores.tolist()
    except Exception:
        traceback.print_exc()
        return "unknown", []

# -----------------------------
# /predict エンドポイント
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "image required"}), 400

        try:
            img = Image.open(BytesIO(request.files["image"].read()))
        except UnidentifiedImageError:
            return jsonify({"error": "cannot identify image"}), 400

        results = {}

        # --- best.onnx 推論 ---
        if session_battery:
            inp_b = preprocess(img, battery_height, battery_width)
            out_b = session_battery.run(None, {input_battery: inp_b})
            label_b, scores_b = postprocess(out_b, ["battery"], threshold=BATTERY_THRESHOLD)
            results["best.onnx"] = {"result": label_b, "scores": scores_b}

        # --- other.onnx 推論 ---
        if session_other:
            inp_o = preprocess(img, other_height, other_width)
            out_o = session_other.run(None, {input_other: inp_o})
            label_o, scores_o = postprocess(out_o, other_names, threshold=OTHER_THRESHOLD)
            results["other.onnx"] = {"result": label_o, "scores": scores_o}

        return jsonify(results)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------------
# 健康チェック用
# -----------------------------
@app.route("/")
def index():
    return jsonify({"status": "running"})

# -----------------------------
# Flask起動
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
