import os
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
import numpy as np
from io import BytesIO
import traceback
import requests

# -----------------------------
# 基本設定
# -----------------------------
PORT = int(os.environ.get("PORT", 10000))
MODEL_PATH = "other.onnx"

# Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_TABLE = os.environ.get("SUPABASE_TABLE", "ai_results")

app = Flask(__name__)
CORS(app)


# -----------------------------
# モデルロード
# -----------------------------
def load_model(path):
    try:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        shape = session.get_inputs()[0].shape
        h, w = shape[2], shape[3]
        print(f"✔ Loaded model: {path} (input={w}x{h})")
        return session, input_name, h, w
    except Exception:
        print("❌ Failed to load ONNX model")
        traceback.print_exc()
        return None, None, None, None


session, input_name, model_h, model_w = load_model(MODEL_PATH)


# -----------------------------
# 前処理
# -----------------------------
def preprocess(img, h, w):
    img = img.convert("RGB")
    img = img.resize((w, h))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, 0)
    return arr


# -----------------------------
# COCO クラス（80クラス）
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
# 後処理（最もスコアが高い1クラスのみ返す）
# -----------------------------
def postprocess(output):
    preds = output[0]  # (num_boxes, 85)

    if preds.size == 0:
        return []

    # objectness, class scores
    obj = preds[:, 4:5]
    cls = preds[:, 5:]
    scores = obj * cls  # combine

    # 最も自信の高いボックスを取得
    max_score_flat = scores.max()
    idx_box, idx_class = np.unravel_index(scores.argmax(), scores.shape)

    best_class = COCO_CLASSES[idx_class]

    return [best_class]


# -----------------------------
# Supabase に保存
# -----------------------------
def save_to_supabase(labels):
    if SUPABASE_URL is None or SUPABASE_KEY is None:
        print("⚠ Supabase 情報が未設定。保存スキップ。")
        return False

    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

    data = {"labels": labels}

    try:
        res = requests.post(url, json=data, headers=headers)
        print("Supabase Response:", res.status_code, res.text)
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
        if "image" not in request.files:
            return jsonify({"error": "image required"}), 400

        try:
            img = Image.open(BytesIO(request.files["image"].read()))
        except UnidentifiedImageError:
            return jsonify({"error": "invalid image"}), 400

        inp = preprocess(img, model_h, model_w)
        out = session.run(None, {input_name: inp})
        labels = postprocess(out)

        # 保存
        save_to_supabase(labels)

        return jsonify({"result": labels})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return jsonify({"status": "running"})


if __name__ == "__main__":
    print("🔥 Flask Inference Server (COCO only, Supabase enabled)")
    app.run(host="0.0.0.0", port=PORT)
