import os
import sys
import time
import json
import logging
from io import BytesIO

import requests
from flask import Flask, request, jsonify
from PIL import Image

# Ultralytics/Yolo import may be heavy; import lazily if needed
try:
    from ultralytics import YOLO
except Exception as e:
    print("ultralytics import error:", e)
    # We'll attempt to fail loudly on model load later

# --- Logging ---
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ai-server")

# --- Environment / defaults ---
PORT = int(os.environ.get("PORT", 10000))
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/best.pt")   # prefer mounted/in-repo path
MODEL_URL = os.environ.get("MODEL_URL")                    # if provided, will download if MODEL_PATH missing
OTHER_MODEL = os.environ.get("OTHER_MODEL", "yolov8n.pt")  # default small model
YOLO_CONFIG_DIR = os.environ.get("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
SUPABASE_URL = os.environ.get("https://njlztbylmzysvfmtwweh.supabase.co")
SUPABASE_API_KEY = os.environ.get("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5qbHp0YnlsbXp5c3ZmbXR3d2VoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTA5NjkwMSwiZXhwIjoyMDc2NjcyOTAxfQ.uUdg3jv-GXSZ9GpC8eULMhW-NxWjCL7VH7kxClaLvkM")
SUPABASE_TABLE = os.environ.get("SUPABASE_TABLE", "ai_results")

# Performance / resource hints
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Ensure YOLO config dir is writable
os.environ["YOLO_CONFIG_DIR"] = YOLO_CONFIG_DIR
os.makedirs(YOLO_CONFIG_DIR, exist_ok=True)

app = Flask(__name__)

# --- helper: download large file with streaming and resume support (basic) ---
def download_file(url: str, dest_path: str, chunk_size: int = 8192, max_retries: int = 3):
    """Download a file via streaming to dest_path. Retries on failure."""
    logger.info("Downloading model from %s -> %s", url, dest_path)
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                tmp_path = dest_path + ".partial"
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp_path, dest_path)
                logger.info("Download complete: %s", dest_path)
                return True
        except Exception as e:
            logger.warning("Download attempt %d failed: %s", attempt, e)
            time.sleep(2 * attempt)
    logger.error("Download failed after %d attempts: %s", max_retries, url)
    return False

# --- prepare model file: use local MODEL_PATH if exists, else download from MODEL_URL if provided ---
def ensure_model_file():
    if os.path.exists(MODEL_PATH):
        logger.info("Model file exists at %s", MODEL_PATH)
        return True
    if MODEL_URL:
        ok = download_file(MODEL_URL, MODEL_PATH)
        if ok:
            return True
    logger.error("Model file not found and MODEL_URL not provided or failed. Expected at %s", MODEL_PATH)
    return False

# --- load models (may be heavy) ---
battery_model = None
other_model = None

def load_models():
    global battery_model, other_model
    # battery model
    if not ensure_model_file():
        raise RuntimeError("No battery model available.")
    try:
        logger.info("Loading battery model from: %s", MODEL_PATH)
        battery_model = YOLO(MODEL_PATH)
        logger.info("Battery model loaded.")
    except Exception as e:
        logger.exception("Failed to load battery model: %s", e)
        raise

    # other model (small default)
    try:
        logger.info("Loading other model: %s", OTHER_MODEL)
        other_model = YOLO(OTHER_MODEL)
        logger.info("Other model loaded.")
    except Exception as e:
        logger.warning("Failed to load other model '%s': %s", OTHER_MODEL, e)
        other_model = None

# --- Supabase push helper ---
def push_to_supabase(class_name: str):
    if not SUPABASE_URL or not SUPABASE_API_KEY:
        logger.debug("Supabase not configured, skipping push.")
        return False
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    headers = {
        "apikey": SUPABASE_API_KEY,
        "Authorization": f"Bearer {SUPABASE_API_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }
    payload = {"class": class_name}
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=5)
        r.raise_for_status()
        logger.info("Pushed to Supabase: %s -> %s", class_name, r.status_code)
        return True
    except Exception as e:
        logger.warning("Supabase push failed: %s", e)
        return False

# --- image loader ---
def load_image_from_file_storage(file_storage):
    # Flask FileStorage
    stream = BytesIO(file_storage.read())
    img = Image.open(stream).convert("RGB")
    return img

# --- prediction helpers ---
def predict_battery(img):
    if battery_model is None:
        return None, 0.0
    # ultralytics model.predict returns a Results object; conf filtering set by conf param
    res = battery_model.predict(img, conf=0.5, verbose=False)
    if not res or len(res) == 0:
        return None, 0.0
    r0 = res[0]
    if hasattr(r0, "boxes") and len(r0.boxes) > 0:
        # We only need boolean detection; optionally compute best confidence
        scores = r0.boxes.conf.tolist() if hasattr(r0.boxes, "conf") else []
        best = max(scores) if scores else 0.0
        return "battery", float(best)
    return None, 0.0

def predict_other(img):
    if other_model is None:
        return None, 0.0
    res = other_model.predict(img, conf=0.3, verbose=False)
    if not res or len(res) == 0:
        return None, 0.0
    r0 = res[0]
    if hasattr(r0, "boxes") and len(r0.boxes) > 0:
        cls_idx = int(r0.boxes.cls[0])
        class_name = other_model.names.get(cls_idx, str(cls_idx)) if hasattr(other_model, "names") else str(cls_idx)
        scores = r0.boxes.conf.tolist() if hasattr(r0.boxes, "conf") else []
        best = max(scores) if scores else 0.0
        return class_name, float(best)
    return None, 0.0

# --- routes ---
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "AI inference server up", "model_path": MODEL_PATH}), 200

@app.route("/health", methods=["GET"])
def health():
    status = {
        "battery_model_loaded": battery_model is not None,
        "other_model_loaded": other_model is not None,
    }
    return jsonify(status), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "image file missing"}), 400
    try:
        img = load_image_from_file_storage(request.files["image"])
    except Exception as e:
        logger.exception("Failed to read image: %s", e)
        return jsonify({"error": "invalid image"}), 400

    # battery first
    try:
        battery_class, battery_conf = predict_battery(img)
    except Exception as e:
        logger.exception("battery prediction error: %s", e)
        battery_class, battery_conf = None, 0.0

    if battery_class:
        push_to_supabase("battery")
        return jsonify({"result": "battery", "confidence": battery_conf}), 200

    # fallback other model
    try:
        other_class, other_conf = predict_other(img)
    except Exception as e:
        logger.exception("other model prediction error: %s", e)
        other_class, other_conf = None, 0.0

    if other_class:
        push_to_supabase(other_class)
        return jsonify({"result": other_class, "confidence": other_conf}), 200

    push_to_supabase("unknown")
    return jsonify({"result": "unknown", "confidence": 0.0}), 200

# --- startup: load models once on boot (fail early if impossible) ---
if __name__ != "__main__":
    # If imported by WSGI server, don't auto-load here; explicit import path will load below
    pass

def start_app():
    # load models and return Flask app
    try:
        load_models()
    except Exception as e:
        logger.exception("Model loading failed on startup: %s", e)
        # Allow server to start, but health will show model missing
    return app

# If run directly (python server.py), load models then run
if __name__ == "__main__":
    start_app()
    logger.info("Starting Flask on 0.0.0.0:%d", PORT)
    app.run(host="0.0.0.0", port=PORT)
