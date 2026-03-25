from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import cv2
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm

from backend.config import settings
from backend.models import SearchResult
from backend.services.color_utils import detect_requested_color, infer_color_from_image, normalize_color

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _tokenize_query(text: str) -> set[str]:
    tokens = []
    for t in (text or "").lower().replace("/", " ").replace("-", " ").split():
        t = t.strip(" ,.;:!?()[]{}\"'")
        if len(t) >= 3:
            tokens.append(t)
    return set(tokens)


class LocalVectorStore:
    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.conn = sqlite3.connect(settings.sqlite_db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

        self.index = None
        self._load_or_init_index()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_type TEXT NOT NULL,
                source_path TEXT NOT NULL,
                preview_path TEXT NOT NULL,
                timestamp_sec REAL,
                title TEXT,
                category TEXT,
                color TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                vector_id INTEGER PRIMARY KEY,
                item_id INTEGER NOT NULL,
                FOREIGN KEY(item_id) REFERENCES items(id)
            )
            """
        )
        self.conn.commit()

    def _table_exists(self, table_name: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return cur.fetchone() is not None

    def _table_columns(self, table_name: str) -> set[str]:
        cur = self.conn.cursor()
        try:
            cur.execute(f"PRAGMA table_info({table_name})")
            rows = cur.fetchall()
        except sqlite3.DatabaseError:
            return set()
        return {r[1] for r in rows}

    def _load_or_init_index(self) -> None:
        if settings.faiss_index_path.exists():
            self.index = faiss.read_index(str(settings.faiss_index_path))
        else:
            self.index = faiss.IndexFlatIP(768)

    def _persist_index(self) -> None:
        faiss.write_index(self.index, str(settings.faiss_index_path))

    @property
    def vectors_count(self) -> int:
        return int(self.index.ntotal if self.index is not None else 0)

    def reset(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM embeddings")
        cur.execute("DELETE FROM items")
        self.conn.commit()
        self.index = faiss.IndexFlatIP(768)
        self._persist_index()

    @staticmethod
    def _iter_files(folder: Path, recursive: bool, exts: set[str]) -> Iterable[Path]:
        pattern = "**/*" if recursive else "*"
        for p in folder.glob(pattern):
            if p.is_file() and p.suffix.lower() in exts:
                yield p

    def _insert_item(
        self,
        media_type: str,
        source_path: str,
        preview_path: str,
        timestamp_sec: float | None,
        title: str,
        category: str,
        color: str,
    ) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO items(media_type, source_path, preview_path, timestamp_sec, title, category, color)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (media_type, source_path, preview_path, timestamp_sec, title, category, color),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def _bind_vector(self, vector_id: int, item_id: int) -> None:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO embeddings(vector_id, item_id) VALUES (?, ?)", (vector_id, item_id))
        self.conn.commit()

    def rebuild_images(self, folder_path: str, recursive: bool = True, clear_existing: bool = False) -> tuple[int, int]:
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        if clear_existing:
            self.reset()
        files = list(self._iter_files(folder, recursive, IMAGE_EXTS))

        indexed_items = 0
        indexed_vectors = 0
        for fp in tqdm(files, desc="Indexing images"):
            try:
                image = Image.open(fp).convert("RGB")
                emb = self.embedder.embed_image(image).astype(np.float32)

                vector_id = self.vectors_count
                self.index.add(emb)

                name = fp.stem.replace("_", " ")
                item_id = self._insert_item(
                    media_type="image",
                    source_path=str(fp),
                    preview_path=str(fp),
                    timestamp_sec=None,
                    title=name,
                    category="fashion",
                    color=normalize_color(name),
                )
                self._bind_vector(vector_id, item_id)

                indexed_items += 1
                indexed_vectors += 1
            except Exception:
                continue

        self._persist_index()
        return indexed_items, indexed_vectors

    def rebuild_videos(
        self,
        folder_path: str,
        recursive: bool = True,
        sample_fps: float | None = None,
        clear_existing: bool = False,
    ) -> tuple[int, int]:
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        if clear_existing:
            self.reset()
        files = list(self._iter_files(folder, recursive, VIDEO_EXTS))
        fps_target = sample_fps or settings.sample_fps

        indexed_items = 0
        indexed_vectors = 0

        for video_path in tqdm(files, desc="Indexing videos"):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue

            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            stride = max(int(round(src_fps / max(fps_target, 0.1))), 1)
            frame_idx = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx % stride != 0:
                    frame_idx += 1
                    continue

                timestamp_sec = float(frame_idx / src_fps)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)

                try:
                    emb = self.embedder.embed_image(pil).astype(np.float32)
                    vector_id = self.vectors_count
                    self.index.add(emb)

                    title = video_path.stem.replace("_", " ")
                    item_id = self._insert_item(
                        media_type="video_frame",
                        source_path=str(video_path),
                        preview_path=str(video_path),
                        timestamp_sec=timestamp_sec,
                        title=title,
                        category="fashion_video",
                        color="unknown",
                    )
                    self._bind_vector(vector_id, item_id)

                    indexed_items += 1
                    indexed_vectors += 1
                except Exception:
                    pass

                frame_idx += 1

            cap.release()

        self._persist_index()
        return indexed_items, indexed_vectors

    def _item_from_vector(self, vector_id: int) -> sqlite3.Row | None:
        cur = self.conn.cursor()

        items_cols = self._table_columns("items")
        has_embeddings = self._table_exists("embeddings")

        # Native schema path (app-managed DB)
        if has_embeddings and {"id", "source_path"}.issubset(items_cols):
            cur.execute(
                """
                SELECT i.*
                FROM embeddings e
                JOIN items i ON e.item_id = i.id
                WHERE e.vector_id = ?
                """,
                (int(vector_id),),
            )
            row = cur.fetchone()
            if row is not None:
                return row

        # Prebuilt VectorDB schema path: items(item_id, file_path)
        if {"item_id", "file_path"}.issubset(items_cols):
            cur.execute(
                "SELECT * FROM items WHERE item_id = ?",
                (int(vector_id),),
            )
            return cur.fetchone()

        return None

    def search(self, query_vec: np.ndarray, top_k: int = 5, text_hint: str | None = None) -> list[SearchResult]:
        if self.vectors_count == 0:
            return []

        k = min(max(top_k, settings.top_k_min), settings.top_k_max)
        D, I = self.index.search(query_vec.astype(np.float32), k=min(k * 4, self.vectors_count))

        requested_color = detect_requested_color(text_hint or "")
        query_tokens = _tokenize_query(text_hint or "")
        out: list[SearchResult] = []

        for score, vector_id in zip(D[0], I[0]):
            if vector_id < 0:
                continue

            row = self._item_from_vector(int(vector_id))
            if row is None:
                continue

            adjusted_score = float(score)

            # Normalize row across supported schemas
            if "id" in row.keys():
                row_id = int(row["id"])
                row_media_type = row["media_type"]
                row_source_path = row["source_path"]
                row_preview_path = row["preview_path"]
                row_timestamp = row["timestamp_sec"]
                row_title = row["title"] or "fashion item"
                row_category = row["category"] or "fashion"
                row_color_raw = row["color"] or "unknown"
            else:
                # prebuilt schema fallback
                row_id = int(row["item_id"])
                row_media_type = "image"
                row_source_path = row["file_path"]
                row_preview_path = row["file_path"]
                row_timestamp = None
                row_title = (
                    (row["display_name"] if "display_name" in row.keys() else None)
                    or (row["article_type"] if "article_type" in row.keys() else None)
                    or Path(row["file_path"]).stem.replace("_", " ")
                )
                row_category = (
                    (row["master_category"] if "master_category" in row.keys() else None)
                    or "fashion"
                )
                row_color_raw = (
                    (row["base_colour"] if "base_colour" in row.keys() else None)
                    or normalize_color(row_title)
                )

                # if no textual color metadata exists (e.g., numeric filenames), infer from image pixels
                if row_color_raw == "unknown" and requested_color is not None:
                    row_color_raw = infer_color_from_image(row_source_path)

            row_color = normalize_color(row_color_raw)

            # light rerank boost for explicit color request
            if requested_color is not None:
                if row_color == requested_color:
                    adjusted_score += 0.03
                elif row_color != "unknown":
                    adjusted_score -= 0.01

            # lexical rerank from metadata when available (helps short text queries)
            if query_tokens and "item_id" in row.keys():
                lexical_fields = [
                    row_title,
                    row_category,
                    (row["article_type"] if "article_type" in row.keys() else ""),
                    (row["usage"] if "usage" in row.keys() else ""),
                    (row["gender"] if "gender" in row.keys() else ""),
                    (row["sub_category"] if "sub_category" in row.keys() else ""),
                ]
                hay = " ".join([str(x).lower() for x in lexical_fields if x])
                overlap = sum(1 for t in query_tokens if t in hay)
                adjusted_score += 0.012 * overlap

            out.append(
                SearchResult(
                    item_id=row_id,
                    media_type=row_media_type,
                    source_path=row_source_path,
                    preview_path=row_preview_path,
                    timestamp_sec=row_timestamp,
                    title=row_title,
                    category=row_category,
                    color=row_color,
                    score=adjusted_score,
                )
            )

        out.sort(key=lambda x: x.score, reverse=True)
        return out[:k]
