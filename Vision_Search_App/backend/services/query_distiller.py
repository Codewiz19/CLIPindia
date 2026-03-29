from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


SYSTEM_PROMPT = (
    "You are a strict Indian-fashion query distiller for retrieval. "
    "Extract compact visual search intent from user text and return JSON only. "
    "Keep culturally relevant fashion terms unchanged when possible. "
    "Do not add explanations."
)


@dataclass
class DistilledQuery:
    query: str
    gender: str
    usage: str
    style_tags: list[str]
    confidence: float


class QueryDistiller:
    def __init__(self, model_id: str, max_new_tokens: int = 24) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._load_failed = False

    @staticmethod
    def _sanitize(text: str) -> str:
        q = (text or "").strip().lower()
        q = re.sub(r"[^a-z0-9\s]", " ", q)
        q = " ".join(q.split())
        return q

    @staticmethod
    def _clip_confidence(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @classmethod
    def _fallback_distilled(cls, raw: str) -> DistilledQuery:
        return DistilledQuery(
            query=raw,
            gender="unknown",
            usage="casual",
            style_tags=[],
            confidence=0.45,
        )

    @classmethod
    def _parse_json_payload(cls, text: str, raw: str) -> DistilledQuery:
        payload_text = (text or "").strip()
        if not payload_text:
            return cls._fallback_distilled(raw)

        # tolerate minor wrapper text from LLM by extracting first JSON object
        start = payload_text.find("{")
        end = payload_text.rfind("}")
        if start >= 0 and end > start:
            payload_text = payload_text[start : end + 1]

        try:
            data = json.loads(payload_text)
        except Exception:
            return cls._fallback_distilled(raw)

        query = cls._sanitize(str(data.get("query", ""))) or raw
        gender = cls._sanitize(str(data.get("gender", "unknown"))) or "unknown"
        usage = cls._sanitize(str(data.get("usage", "casual"))) or "casual"

        tags_in = data.get("style_tags", [])
        tags: list[str] = []
        if isinstance(tags_in, list):
            for t in tags_in:
                ts = cls._sanitize(str(t))
                if ts and ts not in tags:
                    tags.append(ts)

        try:
            conf = cls._clip_confidence(float(data.get("confidence", 0.45)))
        except Exception:
            conf = 0.45

        return DistilledQuery(
            query=query,
            gender=gender,
            usage=usage,
            style_tags=tags[:4],
            confidence=conf,
        )

    def _ensure_loaded(self) -> bool:
        if self._load_failed:
            return False
        if self._tokenizer is not None and self._model is not None:
            return True

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self._model.eval()
            return True
        except Exception:
            self._load_failed = True
            self._tokenizer = None
            self._model = None
            return False

    def is_available(self) -> bool:
        return self._ensure_loaded()

    def distill(self, user_text: str) -> str:
        return self.distill_slots(user_text).query

    def distill_slots(self, user_text: str) -> DistilledQuery:
        raw = self._sanitize(user_text)
        if not raw:
            return self._fallback_distilled("")

        if not self._ensure_loaded():
            return self._fallback_distilled(raw)

        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            return self._fallback_distilled(raw)

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Return strict JSON with keys: query, gender, usage, style_tags, confidence. "
                        "Rules: query should be compact retrieval phrase with key visual terms; "
                        "gender in [women, men, unisex, unknown]; usage in [ethnic wear, casual, formal, party, wedding, sports, unknown]; "
                        "style_tags as short list (max 4). confidence in [0,1]. "
                        f"Input: {raw}"
                    ),
                },
            ]

            if hasattr(tokenizer, "apply_chat_template"):
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = f"System: {SYSTEM_PROMPT}\nUser: {raw}\nAssistant:"

            inputs = tokenizer(prompt, return_tensors="pt")
            output_ids = model.generate(
                **inputs,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

            generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            distilled = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            return self._parse_json_payload(distilled, raw)
        except Exception:
            return self._fallback_distilled(raw)
