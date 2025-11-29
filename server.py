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
BEST_MODEL = "best.onnx"    # バッテリ判定モデル
OTHER_MODEL = "other.onnx"  # YOLOv8 ONNXモデル

# 推論閾値
BATTERY_THRESHOLD = 0.3
OTHER_THRESHOLD = 0.25

# COCOクラス名（other.onnx用）
COCO_NAMES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
              5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
              10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
              14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
              20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
              25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
              30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
              35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
              39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
              45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
              51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair',
              57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet',
              62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
              68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
              73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
              78: 'hair drier', 79: 'toothbrush'}

# Flaskアプリ作成
app = Flask(__name__)

# -----------------------------
# モデルロード関数
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
# Letterboxリサイズ（YOLOv8推奨）
# -----------------------------
def letterbox(img, new_shape=(640,640)):
    w, h = img.size
    nw, nh = new_shape
    scale = min(nw/w, nh/h)
    new_w, new_h = int(w*scale), int(h*scale)
    img_resized = img.resize((new_w,new_h), Image.BILINEAR)
    new_img = Image.new("RGB", new_shape, (114,114,114))
    paste_x, paste_y = (nw-new_w)//2, (nh-new_h)//2
    new_img.paste(img_resized, (paste_x, paste_y))
    arr = np.array(new_img).astype(np.float32)/255.0
    arr = arr.transpose(2,0,1)
    arr = np.expand_dims(arr,0)
    return arr

# -----------------------------
# YOLOv8 ONNX postprocess
# -----------------------------
def postprocess_yolo(output, names, threshold=0.25):
    try:
        preds = output[0]  # shape: [num_boxes, 85]
        boxes, scores, class_ids = [], [], []

        for det in preds:
            obj_score = det[4]
            class_probs = det[5:]
            cls_id = int(np.argmax(class_probs))
            cls_score = class_probs[cls_id] * obj_score
            if cls_score >= threshold:
                boxes.append(det[:4])
                scores.append(cls_score)
                class_ids.append(cls_id)

        if not class_ids:
            return "unknown"

        best_idx = np.argmax(scores)
        return names[class_ids[best_idx]]

    except Exception as e:
        print(e)
        return "unknown"

# -----------------------------
# バッテリ判定 postprocess
# -----------------------------
def postprocess_battery(output, threshold=0.3):
    try:
        preds = output[0][0]
        cls_scores = preds[5:]
        max_score = float(np.max(cls_scores))
        if max_score < threshold:
            return "unknown"
        cls_id = int(np.argmax(cls_scores))
        return "battery" if cls_id==0 else "unknown"
    except Exception:
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
            inp_b = letterbox(img, (battery_width, battery_height))
            out_b = session_battery.run(None, {input_battery: inp_b})
            result_b = postprocess_battery(out_b, BATTERY_THRESHOLD)
            if result_b == "battery":
                return jsonify({"result": result_b})

        # YOLOv8 ONNXモデル
        if session_other:
            inp_o = letterbox(img, (other_width, other_height))
            out_o = session_other.run(None, {input_other: inp_o})
            result_o = postprocess_yolo(out_o, COCO_NAMES, OTHER_THRESHOLD)
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
