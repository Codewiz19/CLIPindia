from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


@app.get("/")
def index():
    return FileResponse(
        static_dir / "index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/media")
def media(path: str = Query(..., description="Absolute file path to image")):
    decoded = unquote(path)
    p = Path(decoded).expanduser().resolve()

    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Image file not found")

    if p.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(status_code=400, detail="Unsupported media type")

    # Keep local-serving safe: only allow files from workspace project root.
    root = settings.project_root.resolve()
    if not str(p).lower().startswith(str(root).lower()):
        raise HTTPException(status_code=403, detail="Path outside project root")

    return FileResponse(p)


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
    use_distiller: bool = Form(True),
):
    if embedder is None or store is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    effective_distiller = bool(use_distiller and embedder.can_use_distiller())
    normalized_text = embedder.normalize_user_query(text, use_distiller=use_distiller)
    text_emb = embedder.embed_text(
        text,
        use_distiller=use_distiller,
        normalized_text=normalized_text,
    )
    results = store.search(text_emb, top_k=top_k, text_hint=normalized_text)
    return SearchResponse(
        query_mode="text",
        top_k=min(max(top_k, 3), 5),
        normalized_text=normalized_text,
        use_distiller_requested=bool(use_distiller),
        use_distiller_effective=effective_distiller,
        results=results,
    )


@app.post("/search/compositional", response_model=SearchResponse)
def search_compositional(
    image: UploadFile = File(...),
    text_intent: str = Form(...),
    top_k: int = Form(settings.default_top_k),
    alpha: float = Form(settings.default_alpha),
    use_distiller: bool = Form(True),
):
    if embedder is None or store is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    effective_distiller = bool(use_distiller and embedder.can_use_distiller())
    alpha = min(max(alpha, 0.0), 1.0)
    pil = _read_image_upload(image)

    image_emb = embedder.embed_image(pil)
    normalized_text = embedder.normalize_user_query(text_intent, use_distiller=use_distiller)
    text_emb = embedder.embed_text(
        text_intent,
        use_distiller=use_distiller,
        normalized_text=normalized_text,
    )
    query_vec = embedder.compose_query(image_emb, text_emb, alpha=alpha)

    results = store.search(query_vec, top_k=top_k, text_hint=normalized_text)
    return SearchResponse(
        query_mode="compositional",
        top_k=min(max(top_k, 3), 5),
        alpha=alpha,
        normalized_text=normalized_text,
        use_distiller_requested=bool(use_distiller),
        use_distiller_effective=effective_distiller,
        results=results,
    )
