#!/usr/bin/env python3
 # -*- coding: utf-8 -*-
 import os
 import traceback
 from flask import Flask, request, jsonify
 from flask_cors import CORS
 from PIL import Image
 import requests
 
 from ultralytics import YOLO  # ★ ここがメイン
 
 # -----------------------------
 # 基本設定
 # -----------------------------
 PORT = int(os.environ.get("PORT", 10000))
 # .onnx でも .pt でもOK（例: "yolov8n.onnx" や "yolov8n.pt"）
 MODEL_PATH = os.environ.get("MODEL_PATH", "other.onnx")
 
 SUPABASE_URL = os.environ.get("SUPABASE_URL")
 SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
 SUPABASE_TABLE = os.environ.get("SUPABASE_TABLE", "ai_results")
 
 app = Flask(__name__)
 CORS(app)
 
 # -----------------------------
 # YOLO モデル読み込み
 # -----------------------------
 try:
     if not os.path.exists(MODEL_PATH):
         raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
     # Ultralytics YOLO にそのまま渡す (.onnx / .pt 両対応)
     yolo_model = YOLO(MODEL_PATH)
     print(f":white_check_mark: YOLO model loaded from {MODEL_PATH}")
 except Exception:
     traceback.print_exc()
     yolo_model = None
     print(":x: Failed to load YOLO model")
 
 # -----------------------------
 # COCOクラス
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
 # Supabase 保存
 # -----------------------------
 def save_to_supabase(class_name: str) -> bool:
     if not SUPABASE_URL or not SUPABASE_KEY:
         print(":警告: Supabase URL/KEY not set. Skipping save.")
         return False
     try:
         url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{SUPABASE_TABLE}"
         headers = {
             "apikey": SUPABASE_KEY,
             "Authorization": f"Bearer {SUPABASE_KEY}",
             "Content-Type": "application/json",
             "Prefer": "return=minimal"
         }
         payload = {"class": class_name}  # 単一クラス名
         res = requests.post(url, json=payload, headers=headers, timeout=10)
         print("Supabase response:", res.status_code, res.text)
         return res.status_code in (200, 201)
     except Exception:
         traceback.print_exc()
         return False
 
 # -----------------------------
 # /predict
 # -----------------------------
 @app.route("/predict", methods=["POST"])
 def predict():
     try:
         # モデルがロードされていない
         if yolo_model is None:
             return jsonify({"error": "model not loaded"}), 500
 
         # 画像がない
         file = request.files.get("image")
         if file is None:
             return jsonify({"error": "image required"}), 400
         if file.filename == "":
             return jsonify({"error": "empty filename"}), 400
 
         # 画像として開く
         try:
             img = Image.open(file.stream).convert("RGB")
         except Exception:
             return jsonify({"error": "invalid image"}), 400
 
         # ★ Ultralytics YOLO にそのまま投げる（前処理・後処理は内部でやってくれる）
         results = yolo_model(img)
         r0 = results[0]
 
         labels = []
 
         # 検出モデルの場合： boxes からクラスとスコアを取り出す
         if r0.boxes is not None and len(r0.boxes) > 0:
             for b in r0.boxes:
                 cls_id = int(b.cls[0].item())
                 conf = float(b.conf[0].item())
 
                 if 0 <= cls_id < len(COCO_CLASSES):
                     cls_name = COCO_CLASSES[cls_id]
                 else:
                     cls_name = str(cls_id)
 
                 labels.append({
                     "class": cls_name,
                     "score": conf
                 })
 
         # Supabase へ保存（検出があった場合のみ）
         saved = False
         if labels:
             for l in labels:
                 save_to_supabase(l["class"])
             saved = True
 
         return jsonify({"result": labels, "saved": saved})
 
     except Exception as e:
         traceback.print_exc()
         return jsonify({"error": str(e)}), 500
 
 # -----------------------------
 # 健康チェック
 # -----------------------------
 @app.route("/", methods=["GET"])
 def index():
     return jsonify({
         "status": "running",
         "model_loaded": (yolo_model is not None),
         "model_path": MODEL_PATH
     })
 
 # -----------------------------
 # メイン
 # -----------------------------
 if __name__ == "__main__":
     print(":fire: Flask Inference Server Ready")
     app.run(host="0.0.0.0", port=PORT)
