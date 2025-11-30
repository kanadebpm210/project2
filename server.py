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
OTHER_MODEL = "other.onnx"  # YOLOv8n ONNX

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
        print(f"Loaded model: {path}, input={height}x{width}")
        return session, input_name, height, width
    except Exception:
        print(f"Failed to load: {path}")
        traceback.print_exc()
        return None, None, None, None

session_other, input_other, other_height, other_width = load_model(OTHER_MODEL)

# -----------------------------
# 画像前処理（YOLO互換）
# -----------------------------
def preprocess(img, target_height, target_width):
    img = img.convert("RGB")
    img = img.resize((target_width, target_height))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = np.expand_dims(arr, 0)
    return arr

# -----------------------------
# ゴミクラス＋傘のみ返す
# -----------------------------
GARBAGE_CLASSES = [
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush', 'umbrella'  # ここに傘を追加
]

# COCO 80 クラス
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# -----------------------------
# 推論後処理（ゴミクラスのみ抽出）
# -----------------------------
def postprocess(output):
    try:
        preds = output[0]  # (num_boxes, 85)
        if preds.size == 0:
            return []

        cls_scores = preds[:, 5:]  # class scores
        cls_ids = np.argmax(cls_scores, axis=1)
        labels = [COCO_CLASSES[i] for i in cls_ids if COCO_CLASSES[i] in GARBAGE_CLASSES]

        # 重複除去
        return list(set(labels))
    except Exception:
        traceback.print_exc()
        return []

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

        if session_other:
            inp = preprocess(img, other_height, other_width)
            out = session_other.run(None, {input_other: inp})
            result = postprocess(out)
            if not result:
                # 空なら一番低いスコアのクラスを返す
                cls_scores_all = out[0][:, 5:]
                if cls_scores_all.size == 0:
                    result = ["unknown"]
                else:
                    cls_max = np.max(cls_scores_all, axis=0)
                    cls_id = int(np.argmax(cls_max))
                    result = [COCO_CLASSES[cls_id]]

            return jsonify({"result": result})

        return jsonify({"result": []})

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
