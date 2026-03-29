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

    vision_onnx_path: Path = project_root / "Ai Engine" / "onnx Model" / "vision.onnx"
    text_onnx_path: Path = project_root / "Ai Engine" / "onnx Model" / "text.onnx"
    processor_dir: Path = project_root / "clipindia_export_bundle" / "hf_processor"

    storage_dir: Path = project_root / "VectorDB Generation"
    faiss_index_path: Path = storage_dir / "catalog.index"
    sqlite_db_path: Path = storage_dir / "catalog.db"

    sample_fps: float = 1.0

    # Optional query distillation stage (PyTorch/Hugging Face)
    enable_query_distiller: bool = True
    query_distiller_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    query_distiller_max_new_tokens: int = 24


settings = Settings()
settings.storage_dir.mkdir(parents=True, exist_ok=True)
