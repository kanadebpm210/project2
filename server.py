import os
import onnxruntime as ort
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import requests
from io import BytesIO

PORT = int(os.environ.get("PORT", 10000))

# File path
BEST_MODEL = "best.onnx"
OTHER_MODEL = "other.onnx"

# Supabase env
SUPABASE_URL  = os.environ.get("SUPABASE_URL")
SUPABASE_API_KEY = os.environ.get("SUPABASE_API_KEY")
SUPABASE_TABLE = os.environ.get("SUPABASE_TABLE", "ai_results")

app = Flask(__name__)

print("Loading best.onnx ...")
session_battery = ort.InferenceSession(BEST_MODEL, providers=["CPUExecutionProvider"])
input_battery = session_battery.get_inputs()[0].name

print("Loading other.onnx ...")
session_other = ort.InferenceSession(OTHER_MODEL, providers=["CPUExecutionProvider"])
input_other = session_other.get_inputs()[0].name


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
        pass


def preprocess(img):
    img = img.resize((640, 640))
    arr = np.array(img).astype(np.float32)
    arr /= 255.0
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, 0)
    return arr


def postprocess(output, names, threshold=0.5):
    preds = output[0][0]
    scores = preds[4]
    cls_scores = preds[5:]

    max_score = np.max(cls_scores)
    cls_id = np.argmax(cls_scores)

    if max_score < threshold:
        return "unknown"

    return names[cls_id] if cls_id < len(names) else "unknown"


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "image required"}), 400

    img = Image.open(BytesIO(request.files["image"].read())).convert("RGB")
    inp = preprocess(img)

    # battery model
    out_b = session_battery.run(None, {input_battery: inp})
    result_b = postprocess(out_b, ["battery"], threshold=0.5)

    if result_b == "battery":
        push_to_supabase("battery")
        return jsonify({"result": "battery"})

    # other model
    other_names = ["plastic", "glass", "paper", "metal", "other"]
    out_o = session_other.run(None, {input_other: inp})
    result_o = postprocess(out_o, other_names, threshold=0.4)

    push_to_supabase(result_o)
    return jsonify({"result": result_o})


@app.route("/")
def index():
    return jsonify({"status": "running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
