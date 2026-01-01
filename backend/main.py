from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import uvicorn
import os

app = FastAPI(title="X-Ray Security API")

# CORS ayarları (React frontend'in erişebilmesi için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Güvenlik için production'da spesifik domain verilmeli
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model yükleme (Yol düzeltmesi)
# Mevcut dosyanın (main.py) bulunduğu klasör: backend/
# Model bir üst klasörde: Xray/best.pt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "best.pt")

model = None

print("="*50)
print(f"🔄 Sistem Başlatılıyor...")
print(f"📂 Çalışma Dizini: {os.getcwd()}")
print(f"📂 Hedef Model Yolu: {MODEL_PATH}")

try:
    if os.path.exists(MODEL_PATH):
        # Dosya bilgilerini göster
        stats = os.stat(MODEL_PATH)
        size_mb = stats.st_size / (1024 * 1024)
        print(f"✅ Model Bulundu! Boyut: {size_mb:.2f} MB")
        
        model = YOLO(MODEL_PATH)
        print(f"🚀 Model Yüklendi: {MODEL_PATH}")
    else:
        print(f"⚠️ {MODEL_PATH} bulunamadı, 'yolo11n.pt' indiriliyor...")
        model = YOLO("yolo11n.pt")
except Exception as e:
    print(f"❌ Model yükleme hatası: {e}")

print("="*50)

@app.get("/")
def read_root():
    return {"status": "active", "model": MODEL_PATH}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), conf: float = 0.25, iou: float = 0.45):
    if model is None:
        raise HTTPException(status_code=500, detail="Model yüklenemedi")
    
    try:
        # Dosyayı oku
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        
        # Inference
        results = model.predict(
            source=img_np,
            conf=conf,
            iou=iou,
            imgsz=640
        )
        
        # Sonuçları formatla
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf_score = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            
            detections.append({
                "class": cls_name,
                "confidence": round(conf_score, 4),
                "bbox": [round(x, 1) for x in xyxy],
                "color": "#FF0000"  # Varsayılan renk, frontend'de özelleştirilebilir
            })
            
        return {
            "detections": detections,
            "count": len(detections),
            "message": "Analiz tamamlandı"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Port 8002 kullanıyoruz (8000 ve 8001 dolu olabilir)
    uvicorn.run(app, host="0.0.0.0", port=8002)

