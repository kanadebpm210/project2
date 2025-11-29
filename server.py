import os
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import traceback

# -----------------------------
# 環境変数 / ポート設定
# -----------------------------
PORT = int(os.environ.get("PORT", 10000))

# モデルファイル
BEST_MODEL = "best.onnx"
OTHER_MODEL = "other.onnx"

# Supabase環境変数
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_API_KEY = os.environ.get("SUPABASE_API_KEY")
SUPABASE_TABLE = os.environ.get("SUPABASE_TABLE", "ai_results")

# Flaskアプリ作成
app = Flask(__name__)

# -----------------------------
# ONNX モデルロード
# -----------------------------
print("Loading best.onnx ...")
try:
    session_battery = ort.InferenceSession(BEST_MODEL, providers=["CPUExecutionProvider"])
    input_battery = session_battery.get_inputs()[0].name
    battery_input_shape = session_battery.get_inputs()[0].shape  # (batch, channels, H, W)
    battery_height, battery_width = battery_input_shape[2], battery_input_shape[3]
except Exception:
    print("Failed to load best.onnx")
    traceback.print_exc()
    session_battery = None

print("Loading other.onnx ...")
try:
    session_other = ort.InferenceSession(OTHER_MODEL, providers=["CPUExecutionProvider"])
    input_other = session_other.get_inputs()[0].name
    other_input_shape = session_other.get_inputs()[0].shape
    other_height, other_width = other_input_shape[2], other_input_shape[3]
except Exception:
    print("Failed to load other.onnx")
    traceback.print_exc()
    session_other = None

# -----------------------------
# Supabaseに結果を送信
# -----------------------------
def push_to_supabase(label):
    if not SUPABASE_URL or not SUPABASE_API_KEY:
        return
    try:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}",
            headers={
                "apikey": SUPABASE_API_KEY,
                "Authorization": f"Bearer {SUPABASE_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"class": label},
            timeout=6,
        )
    except Exception:
        traceback.print_exc()

# -----------------------------
# 画像前処理
# -----------------------------
def preprocess(img, target_height, target_width):
    img = img.resize((target_width, target_height))
    arr = np.array(img).astype(np.float32)
    arr /= 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, 0)  # batch dimension
    return arr

# -----------------------------
# 出力後処理
# -----------------------------
def postprocess(output, names, threshold=0.5):
    preds = output[0][0]
    if len(preds) < 6:
        return "unknown"

    cls_scores = preds[5:]
    max_score = np.max(cls_scores)
    cls_id = np.argmax(cls_scores)

    if max_score < threshold:
        return "unknown"

    return names[cls_id] if cls_id < len(names) else "unknown"

# -----------------------------
# /predict エンドポイント
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "image required"}), 400

        img = Image.open(BytesIO(request.files["image"].read())).convert("RGB")

        # batteryモデル
        if session_battery:
            inp_b = preprocess(img, battery_height, battery_width)
            out_b = session_battery.run(None, {input_battery: inp_b})
            result_b = postprocess(out_b, ["battery"], threshold=0.5)
            if result_b == "battery":
                push_to_supabase("battery")
                return jsonify({"result": "battery"})

        # otherモデル
        if session_other:
            inp_o = preprocess(img, other_height, other_width)
            other_names = ["plastic", "glass", "paper", "metal", "other"]
            out_o = session_other.run(None, {input_other: inp_o})
            result_o = postprocess(out_o, other_names, threshold=0.4)
            push_to_supabase(result_o)
            return jsonify({"result": result_o})

        return jsonify({"result": "unknown"})

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
