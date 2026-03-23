import re

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
    return t or "unknown"
