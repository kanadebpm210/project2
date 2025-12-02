#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Flask inference server for YOLOv8 models.
- Prefer ultralytics.YOLO if installed (accepts .pt/.onnx).
- Falls back to onnxruntime for ONNX models if ultralytics not available.
- Saves detected class names to Supabase REST table (optional; set env vars).
"""

import os
import traceback
import json
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import requests

# Optional libs
try:
    from ultralytics import YOLO  # type: ignore
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

try:
    import onnxruntime as ort  # type: ignore
    HAS_ONNXRUNTIME = True
except Exception:
    HAS_ONNXRUNTIME = False

# -----------------------------
# Config (env)
# -----------------------------
PORT = int(os.environ.get("PORT", 10000))
MODEL_PATH = os.environ.get("MODEL_PATH", "other.onnx")  # default
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_TABLE = os.environ.get("SUPABASE_TABLE", "ai_results")

app = Flask(__name__)
CORS(app)

# -----------------------------
# COCO classes (80)
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

# -----------------------------
# Helpers
# -----------------------------
def to_numpy(x) -> np.ndarray:
    """Convert torch tensor or numpy array or list -> numpy array."""
    try:
        if hasattr(x, "cpu") and hasattr(x, "numpy"):
            return x.cpu().numpy()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.array(x)
    except Exception:
        return np.array(x)

def ensure_label_dict(x) -> Dict[str, Any]:
    """Normalize a detected item to dict with at least 'class' and optional 'score'."""
    if isinstance(x, dict):
        cls = x.get("class", "unknown")
        score = x.get("score", None)
    else:
        # try common conversions
        try:
            arr = to_numpy(x)
            # if it's string-like
            if arr.dtype.kind in ("U", "S"):
                cls = str(arr.flat[0])
                score = None
            elif arr.size == 0:
                cls = "unknown"
                score = None
            else:
                # fallback: string representation
                cls = str(x)
                score = None
        except Exception:
            cls = str(x)
            score = None
    return {"class": str(cls), "score": float(score) if score is not None else None}

# -----------------------------
# Supabase save
# -----------------------------
def save_to_supabase(class_name: str) -> bool:
    """Save a single class name to Supabase REST table. Returns True on success."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("⚠ Supabase not configured. Skipping save.")
        return False
    try:
        url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{SUPABASE_TABLE}"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        payload = {"class": class_name}
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        print("Supabase response:", res.status_code, res.text)
        return int(res.status_code) in (200, 201)
    except Exception:
        traceback.print_exc()
        return False

# -----------------------------
# ONNX helpers (if used)
# -----------------------------
onnx_session = None
onnx_input_name = None
onnx_h = 640
onnx_w = 640

def load_onnx(path: str) -> bool:
    global onnx_session, onnx_input_name, onnx_h, onnx_w
    if not HAS_ONNXRUNTIME:
        print("onnxruntime not installed.")
        return False
    if not os.path.exists(path):
        print("ONNX model not found:", path)
        return False
    try:
        onnx_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = onnx_session.get_inputs()[0]
        onnx_input_name = inp.name
        shape = inp.shape
        # try to extract shape safely
        try:
            if len(shape) >= 4:
                maybe_h = shape[-2]; maybe_w = shape[-1]
                if isinstance(maybe_h, int):
                    onnx_h = int(maybe_h)
                if isinstance(maybe_w, int):
                    onnx_w = int(maybe_w)
        except Exception:
            pass
        print("Loaded ONNX via onnxruntime; input size:", onnx_h, onnx_w)
        return True
    except Exception:
        traceback.print_exc()
        return False

def letterbox_image_pil(img_pil: Image.Image, new_shape=(640, 640), color=(114,114,114)):
    """Letterbox resize returning padded image array, scale, pad_x_left, pad_y_top."""
    img = np.array(img_pil.convert("RGB"))
    h0, w0 = img.shape[:2]
    new_w, new_h = new_shape
    scale = min(new_w / w0, new_h / h0)
    resize_w, resize_h = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(img, (resize_w, resize_h))
    pad_w, pad_h = new_w - resize_w, new_h - resize_h
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, scale, left, top

# -----------------------------
# NMS & postprocess for ONNX outputs
# -----------------------------
def nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    if boxes is None or boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)
    boxes = boxes.astype(np.float32)
    scores = scores.astype(np.float32)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1].astype(int)
    keep = []
    while order.size > 0:
        i = int(order[0]); keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-12)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int32)

def postprocess_onnx(outputs: List[np.ndarray], orig_h: int, orig_w: int,
                     scale: float, pad_x: int, pad_y: int,
                     input_size: int = 640, conf_thres: float = 0.25, iou_thres: float = 0.45) -> List[Dict[str,Any]]:
    try:
        if not outputs:
            return []
        preds = None
        for out in outputs:
            arr = np.array(out)
            if arr.ndim >= 2 and arr.shape[-1] >= 5:
                preds = arr
                break
        if preds is None:
            return []

        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]
        elif preds.ndim >= 4:
            preds = preds.reshape(-1, preds.shape[-1])

        if preds.size == 0 or preds.shape[1] < 5:
            return []

        obj = preds[:, 4:5]
        cls = preds[:, 5:]
        if cls.shape[1] > 1:
            scores_matrix = obj * cls
            best_cls = np.argmax(scores_matrix, axis=1).astype(int)
            best_scores = scores_matrix[np.arange(scores_matrix.shape[0]), best_cls].astype(float)
        else:
            single = cls[:,0]
            if np.all(np.abs(single - np.round(single)) < 1e-6) and np.max(single) < 1000:
                best_cls = single.astype(int)
                best_scores = obj[:,0].astype(float)
            else:
                best_cls = np.zeros((preds.shape[0],), dtype=int)
                best_scores = (obj[:,0] * single).astype(float)

        mask = best_scores >= conf_thres
        if not mask.any():
            return []

        boxes_xywh = preds[:, :4].astype(float)
        if boxes_xywh.max() <= 1.01:
            boxes_xywh = boxes_xywh * input_size

        cx = boxes_xywh[:,0]; cy = boxes_xywh[:,1]; bw = boxes_xywh[:,2]; bh = boxes_xywh[:,3]
        x1 = cx - bw/2; y1 = cy - bh/2; x2 = cx + bw/2; y2 = cy + bh/2
        boxes = np.stack([x1,y1,x2,y2], axis=1)

        # undo padding & scale
        boxes[:, [0,2]] = (boxes[:, [0,2]] - pad_x) / (scale + 1e-12)
        boxes[:, [1,3]] = (boxes[:, [1,3]] - pad_y) / (scale + 1e-12)

        boxes[:, [0,2]] = boxes[:, [0,2]].clip(0, orig_w - 1)
        boxes[:, [1,3]] = boxes[:, [1,3]].clip(0, orig_h - 1)

        boxes = boxes[mask]
        best_scores = best_scores[mask]
        best_cls = best_cls[mask].astype(int)

        detections: List[Dict[str,Any]] = []
        for cls_id in np.unique(best_cls):
            idxs = np.where(best_cls == cls_id)[0].astype(int)
            if idxs.size == 0:
                continue
            cls_boxes = boxes[idxs]
            cls_scores = best_scores[idxs]
            keep = nms_indices(cls_boxes, cls_scores, iou_threshold=iou_thres)
            if keep.size == 0:
                continue
            kept = idxs[keep]
            for k in kept:
                b = boxes[int(k)]
                s = float(best_scores[int(k)])
                c = int(cls_id)
                label = COCO_CLASSES[c] if 0 <= c < len(COCO_CLASSES) else str(c)
                detections.append({"class": label, "score": float(s), "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])]})
        return detections
    except Exception:
        traceback.print_exc()
        return []

# -----------------------------
# Initialize model(s)
# -----------------------------
using_ultralytics = False
yolo_model = None

if HAS_ULTRALYTICS:
    try:
        if not os.path.exists(MODEL_PATH):
            print("Model not found:", MODEL_PATH)
        else:
            yolo_model = YOLO(MODEL_PATH)
            using_ultralytics = True
            print("Loaded model via ultralytics:", MODEL_PATH)
    except Exception:
        traceback.print_exc()
        yolo_model = None
        using_ultralytics = False

# try onnxruntime fallback if not using ultralytics
if not using_ultralytics:
    if HAS_ONNXRUNTIME:
        ok = load_onnx(MODEL_PATH)
        if not ok:
            print("ONNX fallback failed to load model.")
    else:
        print("No ultralytics and no onnxruntime; install packages to run inference.")

print("Server ready. ultralytics:", using_ultralytics, "onnxruntime:", HAS_ONNXRUNTIME)

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "running",
        "using_ultralytics": using_ultralytics,
        "onnxruntime_available": HAS_ONNXRUNTIME,
        "model_path": MODEL_PATH
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # require file
        file = request.files.get("image")
        if file is None:
            return jsonify({"error": "image required"}), 400
        if file.filename == "":
            return jsonify({"error": "empty filename"}), 400

        try:
            img_pil = Image.open(file.stream).convert("RGB")
        except Exception:
            return jsonify({"error": "invalid image"}), 400

        orig_w, orig_h = img_pil.size
        detections: List[Dict[str,Any]] = []

        # Path A: ultralytics
        if using_ultralytics and yolo_model is not None:
            try:
                results = yolo_model(img_pil)
                if len(results) > 0:
                    r0 = results[0]
                    cls_arr = None
                    conf_arr = None
                    # common attributes
                    try:
                        if hasattr(r0, "boxes") and r0.boxes is not None:
                            if hasattr(r0.boxes, "cls"):
                                cls_arr = to_numpy(r0.boxes.cls).ravel()
                            if hasattr(r0.boxes, "conf"):
                                conf_arr = to_numpy(r0.boxes.conf).ravel()
                            if (cls_arr is None or conf_arr is None) and hasattr(r0.boxes, "data"):
                                data = to_numpy(r0.boxes.data)
                                if data.ndim == 2 and data.shape[1] >= 6:
                                    conf_arr = data[:,4].astype(float)
                                    col5 = data[:,5]
                                    if np.all(np.abs(col5 - np.round(col5)) < 1e-6):
                                        cls_arr = col5.astype(int)
                                    else:
                                        cls_arr = np.zeros((data.shape[0],), dtype=int)
                    except Exception:
                        traceback.print_exc()

                    # fallback iterate
                    if (cls_arr is None) or (conf_arr is None):
                        cls_list = []
                        conf_list = []
                        for b in getattr(r0, "boxes", []):
                            try:
                                c = to_numpy(getattr(b, "cls", None)).ravel()[0]
                                s = to_numpy(getattr(b, "conf", None)).ravel()[0]
                            except Exception:
                                try:
                                    c = getattr(b, "cls", None)
                                    s = getattr(b, "conf", None)
                                except Exception:
                                    c, s = None, None
                            if c is None or s is None:
                                continue
                            cls_list.append(int(c))
                            conf_list.append(float(s))
                        if len(cls_list) > 0:
                            cls_arr = np.array(cls_list, dtype=int)
                            conf_arr = np.array(conf_list, dtype=float)

                    if cls_arr is not None and conf_arr is not None and len(cls_arr) == len(conf_arr):
                        for cid, score in zip(cls_arr.astype(int), conf_arr.astype(float)):
                            name = COCO_CLASSES[cid] if 0 <= cid < len(COCO_CLASSES) else str(cid)
                            detections.append({"class": name, "score": float(score)})
            except Exception:
                traceback.print_exc()

        # Path B: onnxruntime fallback
        elif HAS_ONNXRUNTIME and onnx_session is not None:
            try:
                inp_img, scale, pad_x, pad_y = letterbox_image_pil(img_pil, new_shape=(onnx_w, onnx_h))
                tensor = inp_img.astype(np.float32) / 255.0
                tensor = tensor.transpose(2,0,1)[None, ...].astype(np.float32)  # (1,3,H,W)
                outputs = onnx_session.run(None, {onnx_input_name: tensor})
                dets = postprocess_onnx(outputs, orig_h, orig_w, scale, pad_x, pad_y, input_size=max(onnx_h, onnx_w))
                detections.extend(dets)
            except Exception:
                traceback.print_exc()

        # Debug print
        try:
            print("DEBUG: detections:", json.dumps(detections, ensure_ascii=False))
        except Exception:
            print("DEBUG: detections (could not json.dumps)")

        # Normalize and save
        saved_any = False
        saved_details = []
        for item in detections:
            ld = ensure_label_dict(item)
            cls_name = ld["class"]
            ok = False
            try:
                ok = save_to_supabase(cls_name)
            except Exception:
                traceback.print_exc()
                ok = False
            saved_any = saved_any or ok
            saved_details.append({"class": cls_name, "saved": bool(ok), "score": ld.get("score", None)})

        return jsonify({"result": detections, "saved_any": saved_any, "saved_details": saved_details})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("Starting server. MODEL_PATH=", MODEL_PATH)
    print("ultralytics available:", HAS_ULTRALYTICS, "onnxruntime available:", HAS_ONNXRUNTIME)
    app.run(host="0.0.0.0", port=PORT)
