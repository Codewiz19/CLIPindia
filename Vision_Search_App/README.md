# ClothIndiaVisionSearch (MVP)

FastAPI + ONNX + FAISS CPU local search app.

## Features (implemented)
- Image search (`/search/image`)
- Text search (`/search/text`)
- Compositional search (`/search/compositional`) for cases like: **blue kurta image + "same green kurta"**
- Local vector DB with FAISS CPU
- Local metadata DB with SQLite
- Image indexing and video-frame indexing endpoints
- Minimalist local UI at `/`

## Project Structure
- `backend/main.py` - FastAPI app + endpoints
- `backend/services/embedder.py` - ONNX inference (vision + text)
- `backend/services/index_store.py` - FAISS + SQLite + indexing/search logic
- `backend/services/color_utils.py` - color intent helper/rerank
- `backend/static/index.html` - minimalist UI
- `requirements.txt`

## Prerequisites
From project root, make sure these exist:
- `siglip_vision.onnx`
- `siglip_text.onnx`
- `clipindia_export_bundle/hf_processor/`

## Install
```powershell
cd "d:\Professional Projects\Ai Projects\ClipIndia\ClothIndiaVisionSearch"
& "..\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

## Run
```powershell
cd "d:\Professional Projects\Ai Projects\ClipIndia\ClothIndiaVisionSearch"
& "..\.venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:
- `http://127.0.0.1:8000` (UI)
- `http://127.0.0.1:8000/docs` (API docs)

## Index Data
### Images
`POST /index/rebuild-images`
```json
{
  "folder_path": "D:/path/to/images",
  "recursive": true,
  "clear_existing": false
}
```

### Videos (frame index)
`POST /index/rebuild-videos`
```json
{
  "folder_path": "D:/path/to/videos",
  "recursive": true,
  "clear_existing": false
}
```

## Search
- `POST /search/image` (multipart)
- `POST /search/text` (multipart)
- `POST /search/compositional` (multipart)
  - `image`, `text_intent`, `top_k` (3..5), `alpha` (0..1)

## Notes
- Use `clear_existing=false` to append image/video vectors into one shared index.
- Use `clear_existing=true` to reset and rebuild from scratch.
- `alpha=0.65` is default for image+text composition.
- Color text intent gets a light rerank boost.
