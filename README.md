# X-Ray Security Vision

Bu proje, X-Ray görüntülerinde tehdit unsurlarını (silah, bıçak, mermi vb.) tespit etmek için geliştirilmiş yapay zeka tabanlı bir güvenlik sistemidir.

## Özellikler

- **Gerçek Zamanlı Tespit:** YOLOv11 modeli ile hızlı ve hassas nesne tespiti.
- **Video Analizi:** Canlı kamera akışı veya video dosyaları üzerinde analiz yapabilme.
- **Tehdit Algılama:** Silah ve bıçak gibi tehlikeli nesneler için özel uyarı sistemi.
- **Gelişmiş Arayüz:** React ile geliştirilmiş, kullanıcı dostu modern arayüz.
- **API Desteği:** FastAPI tabanlı esnek backend mimarisi.

## Kurulum

### Gereksinimler
- Python 3.8+
- Node.js 16+

### Backend Kurulumu
```bash
pip install -r requirements.txt
python backend/main.py
```

### Frontend Kurulumu
```bash
cd frontend
npm install
npm run dev
```

## Kullanım
Tarayıcıda `http://localhost:5178` adresine giderek sistemi kullanabilirsiniz.

