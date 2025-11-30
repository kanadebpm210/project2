import os
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
import numpy as np
from io import BytesIO
import traceback

# -----------------------------
# 基本設定
# -----------------------------
PORT = int(os.environ.get("PORT", 10000))
MODEL_PATH = "other.onnx"

app = Flask(__name__)
CORS(app)  # Android からアクセス可能に


# -----------------------------
# モデルロード
# -----------------------------
def load_model(path):
    try:
        session = ort.InferenceSession(
            path,
            providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        shape = session.get_inputs()[0].shape  # (1,3,H,W)
        h, w = shape[2], shape[3]
        print(f"✔ Loaded ONNX: {path}, input={w}x{h}")
        return session, input_name, h, w
    except Exception:
        print("❌ Failed to load ONNX")
        traceback.print_exc()
        return None, None, None, None


session_other, input_other, other_h, other_w = load_model(MODEL_PATH)


# -----------------------------
# 前処理
# -----------------------------
def preprocess(img, h, w):
    img = img.convert("RGB")
    img = img.resize((w, h))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = np.expand_dims(arr, 0)
    return arr


# -----------------------------
# COCO クラス一覧
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

# ゴミとして扱うクラス
GARBAGE_CLASSES = set([
    'bottle','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
    'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors',
    'teddy bear','hair drier','toothbrush','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard',
    'surfboard','tennis racket','backpack'
])


# -----------------------------
# 後処理
# -----------------------------
def postprocess(output):
    try:
        preds = output[0]  # shape: (num_boxes, 85)
        if preds.size == 0:
            return []

        # objectness × class score
        obj = preds[:, 4:5]
        cls = preds[:, 5:]
        scores = obj * cls

        cls_ids = np.argmax(scores, axis=1)

        # ゴミクラスだけ抽出
        detected = []
        for cid in cls_ids:
            label = COCO_CLASSES[cid]
            if label in GARBAGE_CLASSES:
                detected.append(label)

        # detected が空 → スコア最大のクラスを返す
        if not detected:
            best_id = int(np.argmax(scores.max(axis=0)))
            detected = [COCO_CLASSES[best_id]]

        return list(set(detected))

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
            return jsonify({"error": "image file required"}), 400

        # 画像読み込み
        try:
            img_bytes = request.files["image"].read()
            img = Image.open(BytesIO(img_bytes))
        except UnidentifiedImageError:
            return jsonify({"error": "invalid image"}), 400

        # モデル未ロード時
        if not session_other:
            return jsonify({"error": "model not loaded"}), 500

        # 推論
        inp = preprocess(img, other_h, other_w)
        out = session_other.run(None, {input_other: inp})
        result = postprocess(out)

        return jsonify({"result": result})

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
# サーバー起動
# -----------------------------
if __name__ == "__main__":
    print("🔥 Flask ONNX Inference Server Starting...")
    app.run(host="0.0.0.0", port=PORT, debug=False)
