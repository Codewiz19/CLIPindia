import argparse
import csv
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def list_images(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    out = []
    for p in root.glob(pattern):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return sorted(out)


def load_styles_metadata(styles_csv: Path | None) -> dict[str, dict[str, str]]:
    if styles_csv is None or not styles_csv.exists():
        return {}

    meta: dict[str, dict[str, str]] = {}
    with open(styles_csv, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row.get("id", "")).strip()
            if not rid:
                continue
            meta[rid] = {
                "article_type": str(row.get("articleType", "") or "").strip(),
                "base_colour": str(row.get("baseColour", "") or "").strip(),
                "gender": str(row.get("gender", "") or "").strip(),
                "usage": str(row.get("usage", "") or "").strip(),
                "display_name": str(row.get("productDisplayName", "") or "").strip(),
                "sub_category": str(row.get("subCategory", "") or "").strip(),
                "master_category": str(row.get("masterCategory", "") or "").strip(),
            }
    return meta


def build_vector_db(
    dataset_dir: Path,
    vision_onnx: Path,
    processor_dir: Path,
    out_dir: Path,
    styles_csv: Path | None,
    batch_size: int,
    recursive: bool,
    provider: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "catalog.index"
    db_path = out_dir / "catalog.db"

    image_paths = list_images(dataset_dir, recursive=recursive)
    if not image_paths:
        raise RuntimeError(f"No images found under: {dataset_dir}")

    processor = AutoProcessor.from_pretrained(str(processor_dir))
    styles_meta = load_styles_metadata(styles_csv)

    available = ort.get_available_providers()
    if provider == "cuda":
        providers = ["CUDAExecutionProvider"]
    elif provider == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    providers = [p for p in providers if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(vision_onnx), providers=providers)
    print("ONNX providers available:", available)
    print("ONNX provider in use:", session.get_providers())
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    vectors_chunks: list[np.ndarray] = []
    processed_paths: list[str] = []

    batch_imgs = []
    batch_paths = []

    def flush_batch() -> None:
        if not batch_imgs:
            return
        enc = processor(images=batch_imgs, return_tensors="pt")
        pixel_values = enc["pixel_values"].detach().cpu().numpy().astype(np.float32)
        vecs = session.run([output_name], {input_name: pixel_values})[0].astype(np.float32)
        vecs = l2_normalize(vecs)
        vectors_chunks.append(vecs)
        processed_paths.extend(batch_paths)
        batch_imgs.clear()
        batch_paths.clear()

    for p in tqdm(image_paths, desc="Embedding images"):
        try:
            img = Image.open(p).convert("RGB")
            batch_imgs.append(img)
            batch_paths.append(str(p))
            if len(batch_imgs) >= batch_size:
                flush_batch()
        except Exception:
            continue

    flush_batch()

    if not vectors_chunks:
        raise RuntimeError("No valid images were embedded.")

    vectors = np.vstack(vectors_chunks).astype(np.float32)
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(index_path))

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS items")
    cur.execute(
        """
        CREATE TABLE items (
            item_id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL UNIQUE,
            article_type TEXT,
            base_colour TEXT,
            gender TEXT,
            usage TEXT,
            display_name TEXT,
            sub_category TEXT,
            master_category TEXT
        )
        """
    )

    item_rows = []
    for i, path_str in enumerate(processed_paths):
        stem = Path(path_str).stem
        m = styles_meta.get(stem, {})
        item_rows.append(
            (
                i,
                path_str,
                m.get("article_type", ""),
                m.get("base_colour", ""),
                m.get("gender", ""),
                m.get("usage", ""),
                m.get("display_name", ""),
                m.get("sub_category", ""),
                m.get("master_category", ""),
            )
        )

    cur.executemany(
        """
        INSERT INTO items(
            item_id, file_path, article_type, base_colour, gender, usage, display_name, sub_category, master_category
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        item_rows,
    )
    conn.commit()
    conn.close()

    print("Vector DB created successfully")
    print(f"Indexed images: {len(processed_paths)}")
    print(f"FAISS index: {index_path}")
    print(f"Metadata DB: {db_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create FAISS vector DB from image dataset")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--vision-onnx", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument(
        "--styles-csv",
        type=Path,
        default=None,
        help="Optional styles.csv for metadata enrichment (article/color/gender/usage/display_name)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("VectorDB"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="ONNX runtime execution provider. Use 'cuda' with onnxruntime-gpu installed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_vector_db(
        dataset_dir=args.dataset_dir,
        vision_onnx=args.vision_onnx,
        processor_dir=args.processor_dir,
        out_dir=args.out_dir,
        styles_csv=args.styles_csv,
        batch_size=args.batch_size,
        recursive=args.recursive,
        provider=args.provider,
    )


