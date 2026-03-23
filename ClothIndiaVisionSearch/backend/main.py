from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from backend.config import settings
from backend.models import HealthResponse, RebuildRequest, RebuildResponse, SearchResponse
from backend.services.embedder import OnnxEmbedder
from backend.services.index_store import LocalVectorStore

embedder: OnnxEmbedder | None = None
store: LocalVectorStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, store
    try:
        embedder = OnnxEmbedder()
        store = LocalVectorStore(embedder)
    except Exception:
        embedder = None
        store = None
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if embedder and store else "degraded",
        vision_model_loaded=bool(embedder and embedder.vision_session),
        text_model_loaded=bool(embedder and embedder.text_session),
        vectors_count=int(store.vectors_count if store else 0),
    )


@app.post("/index/rebuild-images", response_model=RebuildResponse)
def rebuild_images(req: RebuildRequest):
    if store is None:
        raise HTTPException(status_code=500, detail="Store not initialized")

    try:
        items, vectors = store.rebuild_images(req.folder_path, req.recursive, req.clear_existing)
        return RebuildResponse(indexed_items=items, indexed_vectors=vectors, media_type="image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/index/rebuild-videos", response_model=RebuildResponse)
def rebuild_videos(req: RebuildRequest):
    if store is None:
        raise HTTPException(status_code=500, detail="Store not initialized")

    try:
        items, vectors = store.rebuild_videos(req.folder_path, req.recursive, clear_existing=req.clear_existing)
        return RebuildResponse(indexed_items=items, indexed_vectors=vectors, media_type="video_frame")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _read_image_upload(upload: UploadFile) -> Image.Image:
    try:
        return Image.open(upload.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file") from e


@app.post("/search/image", response_model=SearchResponse)
def search_image(
    image: UploadFile = File(...),
    top_k: int = Form(settings.default_top_k),
):
    if embedder is None or store is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    pil = _read_image_upload(image)
    image_emb = embedder.embed_image(pil)
    results = store.search(image_emb, top_k=top_k)
    return SearchResponse(query_mode="image", top_k=min(max(top_k, 3), 5), results=results)


@app.post("/search/text", response_model=SearchResponse)
def search_text(
    text: str = Form(...),
    top_k: int = Form(settings.default_top_k),
):
    if embedder is None or store is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    text_emb = embedder.embed_text(text)
    results = store.search(text_emb, top_k=top_k, text_hint=text)
    return SearchResponse(query_mode="text", top_k=min(max(top_k, 3), 5), results=results)


@app.post("/search/compositional", response_model=SearchResponse)
def search_compositional(
    image: UploadFile = File(...),
    text_intent: str = Form(...),
    top_k: int = Form(settings.default_top_k),
    alpha: float = Form(settings.default_alpha),
):
    if embedder is None or store is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    alpha = min(max(alpha, 0.0), 1.0)
    pil = _read_image_upload(image)

    image_emb = embedder.embed_image(pil)
    text_emb = embedder.embed_text(text_intent)
    query_vec = embedder.compose_query(image_emb, text_emb, alpha=alpha)

    results = store.search(query_vec, top_k=top_k, text_hint=text_intent)
    return SearchResponse(
        query_mode="compositional",
        top_k=min(max(top_k, 3), 5),
        alpha=alpha,
        results=results,
    )
