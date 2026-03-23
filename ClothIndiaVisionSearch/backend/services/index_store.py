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
from backend.services.color_utils import detect_requested_color, normalize_color

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


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
        cur.execute(
            """
            SELECT i.*
            FROM embeddings e
            JOIN items i ON e.item_id = i.id
            WHERE e.vector_id = ?
            """,
            (int(vector_id),),
        )
        return cur.fetchone()

    def search(self, query_vec: np.ndarray, top_k: int = 5, text_hint: str | None = None) -> list[SearchResult]:
        if self.vectors_count == 0:
            return []

        k = min(max(top_k, settings.top_k_min), settings.top_k_max)
        D, I = self.index.search(query_vec.astype(np.float32), k=min(k * 4, self.vectors_count))

        requested_color = detect_requested_color(text_hint or "")
        out: list[SearchResult] = []

        for score, vector_id in zip(D[0], I[0]):
            if vector_id < 0:
                continue

            row = self._item_from_vector(int(vector_id))
            if row is None:
                continue

            adjusted_score = float(score)
            row_color = normalize_color(row["color"])

            # light rerank boost for explicit color request
            if requested_color is not None:
                if row_color == requested_color:
                    adjusted_score += 0.03
                elif row_color != "unknown":
                    adjusted_score -= 0.01

            out.append(
                SearchResult(
                    item_id=int(row["id"]),
                    media_type=row["media_type"],
                    source_path=row["source_path"],
                    preview_path=row["preview_path"],
                    timestamp_sec=row["timestamp_sec"],
                    title=row["title"] or "fashion item",
                    category=row["category"] or "fashion",
                    color=row["color"] or "unknown",
                    score=adjusted_score,
                )
            )

        out.sort(key=lambda x: x.score, reverse=True)
        return out[:k]
