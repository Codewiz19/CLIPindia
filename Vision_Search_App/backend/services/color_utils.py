import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

COLOR_SYNONYMS = {
    "green": ["green", "olive", "bottle green", "mehendi", "emerald", "mint"],
    "blue": ["blue", "navy", "indigo", "sky blue", "teal"],
    "red": ["red", "maroon", "crimson", "burgundy"],
    "yellow": ["yellow", "mustard"],
    "black": ["black"],
    "white": ["white", "off white", "ivory"],
    "pink": ["pink", "rose", "fuchsia"],
    "purple": ["purple", "violet", "lavender"],
    "orange": ["orange", "rust"],
    "brown": ["brown", "tan", "beige", "khaki"],
    "grey": ["grey", "gray", "charcoal"],
}


def detect_requested_color(text: str) -> str | None:
    t = (text or "").lower()
    for canonical, variants in COLOR_SYNONYMS.items():
        for v in variants:
            if re.search(rf"\b{re.escape(v)}\b", t):
                return canonical
    return None


def normalize_color(raw_color: str) -> str:
    t = (raw_color or "unknown").lower().strip()
    for canonical, variants in COLOR_SYNONYMS.items():
        if any(v in t for v in variants):
            return canonical
    return "unknown"


_COLOR_CENTROIDS = {
    "black": np.array([25, 25, 25], dtype=np.float32),
    "white": np.array([235, 235, 235], dtype=np.float32),
    "grey": np.array([128, 128, 128], dtype=np.float32),
    "red": np.array([195, 55, 55], dtype=np.float32),
    "green": np.array([65, 145, 75], dtype=np.float32),
    "blue": np.array([70, 105, 185], dtype=np.float32),
    "yellow": np.array([220, 190, 70], dtype=np.float32),
    "orange": np.array([220, 135, 60], dtype=np.float32),
    "brown": np.array([145, 105, 70], dtype=np.float32),
    "pink": np.array([220, 145, 175], dtype=np.float32),
    "purple": np.array([145, 95, 175], dtype=np.float32),
}


@lru_cache(maxsize=20000)
def infer_color_from_image(file_path: str) -> str:
    try:
        p = Path(file_path)
        if not p.exists():
            return "unknown"

        with Image.open(p).convert("RGB") as img:
            arr = np.asarray(img.resize((64, 64), Image.BILINEAR), dtype=np.float32)

        mean_rgb = arr.reshape(-1, 3).mean(axis=0)

        best_color = "unknown"
        best_dist = float("inf")
        for c, centroid in _COLOR_CENTROIDS.items():
            d = float(np.linalg.norm(mean_rgb - centroid))
            if d < best_dist:
                best_dist = d
                best_color = c

        return best_color
    except Exception:
        return "unknown"
