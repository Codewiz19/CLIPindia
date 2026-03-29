from pydantic import BaseModel, Field
from typing import Optional


class SearchResult(BaseModel):
    item_id: int
    media_type: str
    source_path: str
    preview_path: str
    timestamp_sec: Optional[float] = None
    title: str
    category: str
    color: str
    score: float


class SearchResponse(BaseModel):
    query_mode: str
    top_k: int
    alpha: Optional[float] = None
    normalized_text: Optional[str] = None
    use_distiller_requested: Optional[bool] = None
    use_distiller_effective: Optional[bool] = None
    results: list[SearchResult]


class RebuildRequest(BaseModel):
    folder_path: str = Field(..., description="Folder containing media files")
    recursive: bool = True
    clear_existing: bool = False


class RebuildResponse(BaseModel):
    indexed_items: int
    indexed_vectors: int
    media_type: str


class HealthResponse(BaseModel):
    status: str
    vision_model_loaded: bool
    text_model_loaded: bool
    vectors_count: int
