from __future__ import annotations

import difflib
import re
import sqlite3

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor

from backend.config import settings


class OnnxEmbedder:
    BASE_SPELL_VOCAB = {
        "kurta", "kurti", "kurtis", "saree", "sherwani", "lehenga", "dupatta", "anarkali", "churidar",
        "salwar", "suit", "ethnic", "traditional", "festive", "bridal", "wedding", "casual", "formal",
        "shirt", "tshirt", "tee", "top", "blazer", "jacket", "hoodie", "sweatshirt", "sweater", "jeans",
        "denim", "trouser", "trousers", "pants", "cargo", "chinos", "shorts", "dress", "gown", "skirt",
        "palazzo", "polo", "printed", "plain", "striped", "checked", "cotton", "linen", "silk", "women",
        "woman", "ladies", "female", "men", "man", "gents", "male", "red", "blue", "green", "yellow",
        "black", "white", "pink", "orange", "purple", "brown", "beige", "maroon", "navy", "olive", "grey",
        "gray", "catalog", "fashion", "clothing", "apparel", "ecommerce", "photo", "image", "wear",
    }

    FASHION_TEXT_TEMPLATES = [
        "a product photo of {q}, for {gender}, {usage}",
        "a product photo of {q}",
        "{q}",
        "fashion catalog image: {q}, category fashion, for {gender}",
        "indian fashion item: {q}, {usage}, style everyday style",
        "an ecommerce catalog photo of {q}",
    ]

    def __init__(self) -> None:
        self.processor = AutoProcessor.from_pretrained(str(settings.processor_dir))

        providers = ["CPUExecutionProvider"]
        self.vision_session = ort.InferenceSession(str(settings.vision_onnx_path), providers=providers)
        self.text_session = ort.InferenceSession(str(settings.text_onnx_path), providers=providers)

        self.vision_input_name = self.vision_session.get_inputs()[0].name
        self.text_input_names = [i.name for i in self.text_session.get_inputs()]
        self.spell_vocab = self._build_spell_vocab()
        self.spell_vocab_list = sorted(self.spell_vocab)

    def _build_spell_vocab(self) -> set[str]:
        vocab = set(self.BASE_SPELL_VOCAB)

        # Extend vocabulary from catalog metadata when available.
        try:
            if settings.sqlite_db_path.exists():
                conn = sqlite3.connect(settings.sqlite_db_path)
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()

                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='items'")
                if cur.fetchone() is not None:
                    cur.execute("PRAGMA table_info(items)")
                    cols = {r[1] for r in cur.fetchall()}

                    text_cols = [
                        c
                        for c in ["title", "category", "color", "display_name", "article_type", "base_colour", "sub_category", "master_category", "gender", "usage"]
                        if c in cols
                    ]

                    for col in text_cols:
                        cur.execute(f"SELECT {col} FROM items WHERE {col} IS NOT NULL LIMIT 20000")
                        for row in cur.fetchall():
                            value = (row[0] or "").lower()
                            for t in re.findall(r"[a-z]+", value):
                                if len(t) >= 3:
                                    vocab.add(t)

                conn.close()
        except Exception:
            # Keep spell-correction non-blocking.
            pass

        return vocab

    def _correct_spelling(self, text: str) -> str:
        tokens = re.findall(r"[a-z]+|[^a-z\s]+", text.lower())
        corrected: list[str] = []

        for tok in tokens:
            if not tok.isalpha() or len(tok) < 4:
                corrected.append(tok)
                continue

            if tok in self.spell_vocab:
                corrected.append(tok)
                continue

            # Conservative nearest-token correction to reduce bad rewrites.
            match = difflib.get_close_matches(tok, self.spell_vocab_list, n=1, cutoff=0.84)
            corrected.append(match[0] if match else tok)

        out = " ".join(corrected)
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)
        out = " ".join(out.split())
        return out

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return v / norms

    def embed_image(self, image: Image.Image) -> np.ndarray:
        enc = self.processor(images=image.convert("RGB"), return_tensors="pt")
        pixel_values = enc["pixel_values"].detach().cpu().numpy().astype(np.float32)
        out = self.vision_session.run(None, {self.vision_input_name: pixel_values})[0]
        return self._normalize(out)

    @staticmethod
    def _strip_query(text: str) -> str:
        q = (text or "").strip().lower()
        if not q:
            return "fashion clothing"

        # reduce compositional filler words that hurt pure semantics
        for token in ["same", "similar", "like", "as", "the", "this", "that"]:
            q = q.replace(f" {token} ", " ")
        q = " ".join(q.split())
        return q or "fashion clothing"

    def normalize_user_query(self, text: str) -> str:
        stripped = self._strip_query(text)
        corrected = self._correct_spelling(stripped)
        return corrected or "fashion clothing"

    def _text_variants(self, text: str) -> list[str]:
        q = self.normalize_user_query(text)
        ql = f" {q} "

        explicit_gender = "women" if " women " in ql or " woman " in ql or " ladies " in ql else (
            "men" if " men " in ql or " man " in ql or " gents " in ql else None
        )

        female_ethnic_tokens = [" kurti ", " saree ", " lehenga ", " dupatta ", " anarkali "]
        male_tokens = [" shirt ", " tshirt ", " t-shirt ", " trousers ", " blazer "]

        if explicit_gender is not None:
            genders = [explicit_gender]
        elif any(t in ql for t in female_ethnic_tokens):
            genders = ["women", "men"]
        elif any(t in ql for t in male_tokens):
            genders = ["men", "women"]
        else:
            genders = ["women", "men"]

        usage = "ethnic wear" if any(t in ql for t in [" kurta ", " kurti ", " saree ", " lehenga ", " sherwani "]) else "casual"

        variants = []
        for gender in genders:
            variants.extend([tmpl.format(q=q, gender=gender, usage=usage) for tmpl in self.FASHION_TEXT_TEMPLATES])

        # keep order while de-duplicating
        return list(dict.fromkeys(variants))

    def embed_text(self, text: str) -> np.ndarray:
        variants = self._text_variants(text)
        enc = self.processor(text=variants, return_tensors="pt", truncation=True, padding=True)

        input_feed = {}
        for name in self.text_input_names:
            if name in enc:
                input_feed[name] = enc[name].detach().cpu().numpy().astype(np.int64)

        if "attention_mask" in self.text_input_names and "attention_mask" not in input_feed:
            input_feed["attention_mask"] = np.ones_like(
                enc["input_ids"].detach().cpu().numpy(), dtype=np.int64
            )

        out = self.text_session.run(None, input_feed)[0].astype(np.float32)
        out = self._normalize(out)

        # weighted ensemble: emphasize first canonical (training-style) variant.
        n = out.shape[0]
        if n == 1:
            return out

        weights = np.full((n,), 0.55 / (n - 1), dtype=np.float32)
        weights[0] = 0.45
        mixed = np.sum(out * weights[:, None], axis=0, keepdims=True)
        return self._normalize(mixed)

    def compose_query(self, image_emb: np.ndarray, text_emb: np.ndarray, alpha: float) -> np.ndarray:
        q = alpha * image_emb + (1.0 - alpha) * text_emb
        return self._normalize(q)
