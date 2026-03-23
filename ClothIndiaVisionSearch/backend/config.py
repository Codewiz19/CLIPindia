from pathlib import Path
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = "ClothIndia Vision Search"
    top_k_min: int = 3
    top_k_max: int = 5
    default_top_k: int = 5
    default_alpha: float = 0.65

    base_dir: Path = Path(__file__).resolve().parent.parent
    project_root: Path = base_dir.parent

    vision_onnx_path: Path = project_root / "siglip_vision.onnx"
    text_onnx_path: Path = project_root / "siglip_text.onnx"
    processor_dir: Path = project_root / "clipindia_export_bundle" / "hf_processor"

    storage_dir: Path = base_dir / "storage"
    faiss_index_path: Path = storage_dir / "catalog.index"
    sqlite_db_path: Path = storage_dir / "catalog.db"

    sample_fps: float = 1.0


settings = Settings()
settings.storage_dir.mkdir(parents=True, exist_ok=True)
