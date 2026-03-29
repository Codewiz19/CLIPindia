from __future__ import annotations

import difflib
import re
import sqlite3

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor

from backend.config import settings
from backend.services.query_distiller import DistilledQuery, QueryDistiller


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
        # Indian-fashion specific protected terms
        "chinon", "chiffon", "banarasi", "bandhani", "kanjivaram", "zari", "zardozi", "gota",
        "patiala", "ghagra", "blouse", "pallu", "kalamkari", "ajrakh", "paithani", "leheriya",
        "scallop", "embroidered", "embroidery",
    }

    FASHION_TEXT_TEMPLATES = [
        "a product photo of {q}, for {gender}, {usage}",
        "a product photo of {q}",
        "{q}",
        "fashion catalog image: {q}, category fashion, for {gender}",
        "indian fashion item: {q}, {usage}, style everyday style",
        "an ecommerce catalog photo of {q}",
    ]

    # Confidence-aware hybrid blend when distiller is enabled.
    DISTILLER_BLEND_LEGACY_MIN = 0.55
    DISTILLER_BLEND_LEGACY_MAX = 0.85
    DISTILLER_BLEND_DISTILLED_MIN = 0.15
    DISTILLER_BLEND_DISTILLED_MAX = 0.45

    DISTILLER_TEMPLATES = [
        "a product photo of {q}, for {gender}, {usage}",
        "a product photo of {q}",
        "indian fashion item: {q}, {usage}, style {style}",
        "{q}",
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
        self.query_distiller: QueryDistiller | None = None
        if settings.enable_query_distiller:
            self.query_distiller = QueryDistiller(
                model_id=settings.query_distiller_model_id,
                max_new_tokens=settings.query_distiller_max_new_tokens,
            )

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
            if not match:
                corrected.append(tok)
                continue

            cand = match[0]
            # Extra guardrails for domain safety.
            if cand[:1] != tok[:1] or abs(len(cand) - len(tok)) > 2:
                corrected.append(tok)
                continue

            corrected.append(cand)

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

    def can_use_distiller(self) -> bool:
        if self.query_distiller is None:
            return False
        try:
            return self.query_distiller.is_available()
        except Exception:
            return False

    def normalize_user_query(self, text: str, use_distiller: bool = True) -> str:
        q = (text or "").strip().lower()
        if use_distiller and self.can_use_distiller():
            q = self.query_distiller.distill(q)

        stripped = self._strip_query(q)
        corrected = self._correct_spelling(stripped)
        return corrected or "fashion clothing"

    def _distill_query(self, text: str) -> DistilledQuery:
        raw = (text or "").strip().lower()
        if not self.can_use_distiller() or self.query_distiller is None:
            normalized = self.normalize_user_query(raw, use_distiller=False)
            return DistilledQuery(
                query=normalized,
                gender="unknown",
                usage="casual",
                style_tags=[],
                confidence=0.0,
            )

        info = self.query_distiller.distill_slots(raw)
        normalized_query = self._correct_spelling(self._strip_query(info.query))
        return DistilledQuery(
            query=normalized_query or "fashion clothing",
            gender=(info.gender or "unknown"),
            usage=(info.usage or "casual"),
            style_tags=info.style_tags,
            confidence=max(0.0, min(1.0, float(info.confidence))),
        )

    def _embed_single_text(self, text: str) -> np.ndarray:
        enc = self.processor(text=[text], return_tensors="pt", truncation=True, padding=True)

        input_feed: dict[str, np.ndarray] = {}
        for name in self.text_input_names:
            if name in enc:
                input_feed[name] = enc[name].detach().cpu().numpy().astype(np.int64)

        if "attention_mask" in self.text_input_names and "attention_mask" not in input_feed:
            input_feed["attention_mask"] = np.ones_like(
                enc["input_ids"].detach().cpu().numpy(), dtype=np.int64
            )

        out = self.text_session.run(None, input_feed)[0].astype(np.float32)
        return self._normalize(out)

    def _text_variants(self, text: str, use_distiller: bool = True) -> list[str]:
        q = self.normalize_user_query(text, use_distiller=use_distiller)
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

    def _embed_text_variants(self, text: str) -> np.ndarray:
        variants = self._text_variants(text, use_distiller=False)
        enc = self.processor(text=variants, return_tensors="pt", truncation=True, padding=True)

        input_feed: dict[str, np.ndarray] = {}
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

    def _distiller_variants(self, info: DistilledQuery) -> list[str]:
        q = info.query or "fashion clothing"
        gender = info.gender if info.gender in {"women", "men", "unisex"} else "unisex"
        usage = info.usage if info.usage else "casual"
        style = ", ".join(info.style_tags) if info.style_tags else "everyday style"

        variants = [t.format(q=q, gender=gender, usage=usage, style=style) for t in self.DISTILLER_TEMPLATES]
        return list(dict.fromkeys(variants))

    def _embed_distiller_variants(self, info: DistilledQuery) -> np.ndarray:
        variants = self._distiller_variants(info)
        enc = self.processor(text=variants, return_tensors="pt", truncation=True, padding=True)

        input_feed: dict[str, np.ndarray] = {}
        for name in self.text_input_names:
            if name in enc:
                input_feed[name] = enc[name].detach().cpu().numpy().astype(np.int64)

        if "attention_mask" in self.text_input_names and "attention_mask" not in input_feed:
            input_feed["attention_mask"] = np.ones_like(
                enc["input_ids"].detach().cpu().numpy(), dtype=np.int64
            )

        out = self.text_session.run(None, input_feed)[0].astype(np.float32)
        out = self._normalize(out)
        weights = np.array([0.40, 0.30, 0.20, 0.10], dtype=np.float32)[: out.shape[0]]
        weights = weights / np.sum(weights)
        mixed = np.sum(out * weights[:, None], axis=0, keepdims=True)
        return self._normalize(mixed)

    def embed_text(
        self,
        text: str,
        use_distiller: bool = True,
        normalized_text: str | None = None,
    ) -> np.ndarray:
        if use_distiller and self.can_use_distiller():
            info = self._distill_query(text)
            if normalized_text:
                info = DistilledQuery(
                    query=normalized_text,
                    gender=info.gender,
                    usage=info.usage,
                    style_tags=info.style_tags,
                    confidence=info.confidence,
                )

            # 1) Distilled template-aligned embedding (new path)
            distilled_emb = self._embed_distiller_variants(info)

            # 2) Legacy template-ensemble embedding (stable baseline)
            legacy_emb = self._embed_text_variants(text)

            # 3) Confidence-aware blend (legacy-dominant when confidence is low)
            c = max(0.0, min(1.0, float(info.confidence)))
            w_distilled = self.DISTILLER_BLEND_DISTILLED_MIN + (
                self.DISTILLER_BLEND_DISTILLED_MAX - self.DISTILLER_BLEND_DISTILLED_MIN
            ) * c
            w_legacy = 1.0 - w_distilled
            w_legacy = max(self.DISTILLER_BLEND_LEGACY_MIN, min(self.DISTILLER_BLEND_LEGACY_MAX, w_legacy))
            total = w_legacy + w_distilled
            w_legacy /= total
            w_distilled /= total

            hybrid = (
                w_legacy * legacy_emb
                + w_distilled * distilled_emb
            )
            return self._normalize(hybrid)

        return self._embed_text_variants(text)

    def compose_query(self, image_emb: np.ndarray, text_emb: np.ndarray, alpha: float) -> np.ndarray:
        q = alpha * image_emb + (1.0 - alpha) * text_emb
        return self._normalize(q)
