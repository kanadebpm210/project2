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

# YOLOv8 ONNXモデル
OTHER_MODEL = "other.onnx"

# Flaskアプリ作成
app = Flask(__name__)

# -----------------------------
# ONNXモデルロード
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
# COCO 80クラス名
# -----------------------------
COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella',
    'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite',
    'baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
    'wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
    'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors',
    'teddy bear','hair drier','toothbrush'
]

# -----------------------------
# ゴミクラス一覧（返したいものだけ）
# -----------------------------
GARBAGE_CLASSES = [
    "bottle","cup","fork","knife","spoon","bowl","banana","apple","sandwich",
    "orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# -----------------------------
# 画像前処理
# -----------------------------
def preprocess(img, target_height, target_width):
    img = img.convert("RGB")
    img = img.resize((target_width, target_height))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = np.expand_dims(arr, 0)
    return arr

# -----------------------------
# 出力後処理
# -----------------------------
def postprocess(output):
    """
    複数物体が写っている場合はゴミクラスを全て返す。
    ゴミがなければスコアが最も高いクラスを1つ返す。
    """
    try:
        preds = output[0]  # (num_boxes, 85)
        if preds.size == 0:
            return [COCO_CLASSES[0]]  # 安全策

        cls_scores = preds[:, 5:]  # class scores
        cls_ids = np.argmax(cls_scores, axis=1)
        detected_classes = [COCO_CLASSES[i] for i in cls_ids]

        # ゴミクラスだけ抽出
        garbage_detected = list({c for c in detected_classes if c in GARBAGE_CLASSES})

        if len(garbage_detected) == 0:
            # ゴミが無ければスコア最大のクラスを1つ返す
            max_scores_per_box = np.max(cls_scores, axis=1)
            top_idx = int(np.argmax(max_scores_per_box))
            top_cls_id = int(np.argmax(cls_scores[top_idx]))
            return [COCO_CLASSES[top_cls_id]]

        return garbage_detected
    except Exception:
        traceback.print_exc()
        return [COCO_CLASSES[0]]  # 安全策

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

        inp = preprocess(img, other_height, other_width)
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
# Flask起動
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
