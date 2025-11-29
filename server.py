# server.py
import os
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
import numpy as np
from io import BytesIO
import traceback

PORT = int(os.environ.get("PORT", 10000))
BEST_MODEL = "best.onnx"      # バッテリー判定
OTHER_MODEL = "other.onnx"    # YOLOv8n ONNX

# COCO 80クラス
CLASS_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
    'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote',
    'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book',
    'clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

app = Flask(__name__)

# ONNX モデルロード
def load_model(path):
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        shape = sess.get_inputs()[0].shape
        return sess, input_name, shape
    except Exception:
        print(f"Failed to load {path}")
        traceback.print_exc()
        return None, None, None

session_battery, input_battery, shape_battery = load_model(BEST_MODEL)
session_other, input_other, shape_other = load_model(OTHER_MODEL)

# 前処理
def preprocess(img, shape):
    img = img.convert("RGB")
    _, _, h, w = shape
    img = img.resize((w, h))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2,0,1)
    arr = np.expand_dims(arr,0)
    return arr

# 推論
def run_battery(img):
    if session_battery is None:
        return "unknown"
    inp = preprocess(img, shape_battery)
    out = session_battery.run(None, {input_battery: inp})
    # 出力次第で 0=unknown, 1=battery など
    pred = np.argmax(out[0], axis=1)[0]
    if pred == 1:
        return "battery"
    else:
        return "unknown"

def run_other(img):
    if session_other is None:
        return "unknown"
    inp = preprocess(img, shape_other)
    out = session_other.run(None, {input_other: inp})
    preds = out[0]
    if len(preds.shape) == 3:
        scores = preds[0,:,5:]
        max_idx = np.argmax(scores)
        class_id = max_idx % len(CLASS_NAMES)
        return CLASS_NAMES[class_id]
    return "unknown"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "image required"}), 400

    try:
        img = Image.open(BytesIO(request.files["image"].read()))
    except UnidentifiedImageError:
        return jsonify({"error": "cannot identify image"}), 400

    # まずバッテリー判定
    label = run_battery(img)
    if label != "unknown":
        return jsonify({"result": label})

    # バッテリーでなければ YOLOv8n 判定
    label = run_other(img)
    return jsonify({"result": label})

@app.route("/")
def index():
    return jsonify({"status":"running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
