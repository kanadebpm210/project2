from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import os
import requests

app = Flask(__name__)

MODEL_PATH = "./best.pt"
battery_model = YOLO(MODEL_PATH)
other_model = YOLO("yolov8m.pt")
MODEL_URL = os.environ.get("MODEL_URL")


if MODEL_URL and not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("Model downloaded!")
SUPABASE_URL = os.environ.get("https://njlztbylmzysvfmtwweh.supabase.co")
SUPABASE_API_KEY = os.environ.get("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5qbHp0YnlsbXp5c3ZmbXR3d2VoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTA5NjkwMSwiZXhwIjoyMDc2NjcyOTAxfQ.uUdg3jv-GXSZ9GpC8eULMhW-NxWjCL7VH7kxClaLvkM")
SUPABASE_TABLE = os.environ.get("SUPABASE_TABLE", "ai_results")

def load_image(file):
    return Image.open(BytesIO(file.read())).convert("RGB")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error":"image missing"}), 400

    img = load_image(request.files["image"])
    r1 = battery_model.predict(img, conf=0.5)[0]

    if len(r1.boxes):
        return jsonify({"result": "battery"})

    r2 = other_model.predict(img, conf=0.3)[0]
    if len(r2.boxes):
        cls = other_model.names[int(r2.boxes.cls[0])]
        return jsonify({"result": cls})

    return jsonify({"result":"unknown"})

@app.route("/")
def home():
    return "AI server is running."

if __name__ == "__main__":
    app.run(port=10000)
