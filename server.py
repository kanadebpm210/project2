import os
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
import numpy as np
from io import BytesIO
import traceback

PORT = int(os.environ.get("PORT", 10000))
OTHER_MODEL = "other.onnx"  # YOLOv8n

app = Flask(__name__)

# -----------------------------
# YOLOv8nモデルロード
# -----------------------------
def load_model(path):
    try:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape  # (batch, channels, H, W)
        height, width = input_shape[2], input_shape[3]
        print(f"Loaded model: {path}, input={height}x{width}")
        return session, input_name, height, width
    except Exception:
        traceback.print_exc()
        return None, None, None, None

session_other, input_other, other_height, other_width = load_model(OTHER_MODEL)

# -----------------------------
# 前処理
# -----------------------------
def preprocess(img, target_height, target_width):
    img = img.convert("RGB")
    img = img.resize((target_width, target_height))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, 0)
    return arr

# -----------------------------
# 後処理（ゴミとして扱うクラスだけ返す）
# -----------------------------
def postprocess(output, garbage_names):
    try:
        preds = output[0]  # (num_boxes, 85)
        if preds.size == 0:
            return "none"

        cls_scores = preds[:, 5:]
        cls_id = int(np.argmax(np.max(cls_scores, axis=0)))

        # COCOクラス名
        other_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
            25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
            39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
            45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
            55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
            60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
            65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
            70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
            75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }

        class_name = other_names.get(cls_id, "none")
        if class_name in garbage_names:
            return class_name
        return "none"

    except Exception:
        traceback.print_exc()
        return "none"

# ゴミとして扱うクラス
GARBAGE_CLASSES = [
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "backpack", "handbag", "suitcase",
    "tie", "umbrella"
]

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
            return jsonify({"error": "cannot identify image"}), 400

        if session_other:
            inp_o = preprocess(img, other_height, other_width)
            out_o = session_other.run(None, {input_other: inp_o})
            result_o = postprocess(out_o, GARBAGE_CLASSES)
            return jsonify({"result": result_o})

        return jsonify({"result": "none"})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return jsonify({"status": "running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
