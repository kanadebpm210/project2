import os
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
import numpy as np
from io import BytesIO
import traceback

# -----------------------------
# 環境変数 / ポート
# -----------------------------
PORT = int(os.environ.get("PORT", 10000))
MODEL_PATH = "yolov8n.onnx"  # 事前に固定サイズでエクスポート済みONNX

# ゴミとして扱うクラス名（COCO 80クラスの例）
GARBAGE_CLASSES = [
    'bottle','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake',
    'chair','couch','potted plant','dining table','toilet','tv','laptop',
    'mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

# COCO 80クラス
COCO_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
    'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
    'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave',
    'oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
    'toothbrush'
]

# -----------------------------
# Flask
# -----------------------------
app = Flask(__name__)

# -----------------------------
# ONNX モデルロード
# -----------------------------
try:
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape  # [1,3,640,640]
    HEIGHT, WIDTH = input_shape[2], input_shape[3]
    print(f"Loaded model: {MODEL_PATH}, input shape: {HEIGHT}x{WIDTH}")
except Exception:
    print("Failed to load ONNX model")
    traceback.print_exc()
    session = None

# -----------------------------
# 前処理
# -----------------------------
def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((WIDTH, HEIGHT))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, 0)   # batch
    return arr

# -----------------------------
# 推論後処理
# -----------------------------
def postprocess(outputs):
    try:
        preds = outputs[0]  # (num_boxes, 85)
        if preds.size == 0:
            return [COCO_NAMES[int(np.argmin(np.random.rand(len(COCO_NAMES))))]]  # 適当に最低スコアのもの

        cls_scores = preds[:, 5:]  # クラススコア
        cls_ids = np.argmax(cls_scores, axis=1)  # 各ボックスの最大スコアクラス
        results = []

        for cid in cls_ids:
            cls_name = COCO_NAMES[cid]
            if cls_name in GARBAGE_CLASSES and cls_name not in results:
                results.append(cls_name)

        # もしゴミクラスが1つも無ければ、スコア最小のもの1つ返す
        if not results:
            cls_max = np.max(cls_scores, axis=0)
            cid = int(np.argmin(cls_max))
            results.append(COCO_NAMES[cid])

        return results
    except Exception:
        traceback.print_exc()
        return ["bottle"]  # デフォルト1つ返す

# -----------------------------
# /predict
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error":"image required"}), 400
    try:
        img = Image.open(BytesIO(request.files["image"].read()))
    except UnidentifiedImageError:
        return jsonify({"error":"cannot identify image"}), 400

    inp = preprocess(img)
    outputs = session.run(None, {input_name: inp})
    result = postprocess(outputs)
    return jsonify({"result": result})

# -----------------------------
# 健康チェック
# -----------------------------
@app.route("/")
def index():
    return jsonify({"status":"running"})

# -----------------------------
# Flask 起動
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
