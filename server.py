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

# モデルファイル
BEST_MODEL = "best.onnx"
OTHER_MODEL = "other.onnx"

# Flaskアプリ作成
app = Flask(__name__)

# -----------------------------
# ONNX モデルロード
# -----------------------------
def load_model(path):
    try:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape  # (batch, channels, H, W)
        height, width = input_shape[2], input_shape[3]
        return session, input_name, height, width
    except Exception:
        print(f"Failed to load {path}")
        traceback.print_exc()
        return None, None, None, None

session_battery, input_battery, battery_height, battery_width = load_model(BEST_MODEL)
session_other, input_other, other_height, other_width = load_model(OTHER_MODEL)

# -----------------------------
# COCOクラス名（YOLOv8n ONNX用）
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
# YOLO 出力後処理
# -----------------------------
def postprocess_yolo(output):
    try:
        preds = output[0][0]
        if len(preds) < 6:
            return "unknown"
        obj_score = float(preds[4])
        cls_scores = preds[5:]
        cls_id = int(np.argmax(cls_scores))
        label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "unknown"
        return label
    except Exception:
        traceback.print_exc()
        return "unknown"

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

        # batteryモデル
        if session_battery:
            inp_b = preprocess(img, battery_height, battery_width)
            out_b = session_battery.run(None, {input_battery: inp_b})
            # 最大スコアで battery を返す
            result_b = "battery" if np.max(out_b[0]) > 0 else "unknown"
            if result_b == "battery":
                return jsonify({"result": result_b})

        # otherモデル
        if session_other:
            inp_o = preprocess(img, other_height, other_width)
            out_o = session_other.run(None, {input_other: inp_o})
            result_o = postprocess_yolo(out_o)
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
