from __future__ import annotations

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor

from backend.config import settings


class OnnxEmbedder:
    def __init__(self) -> None:
        self.processor = AutoProcessor.from_pretrained(str(settings.processor_dir))

        providers = ["CPUExecutionProvider"]
        self.vision_session = ort.InferenceSession(str(settings.vision_onnx_path), providers=providers)
        self.text_session = ort.InferenceSession(str(settings.text_onnx_path), providers=providers)

        self.vision_input_name = self.vision_session.get_inputs()[0].name
        self.text_input_names = [i.name for i in self.text_session.get_inputs()]

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return v / norms

    def embed_image(self, image: Image.Image) -> np.ndarray:
        enc = self.processor(images=image.convert("RGB"), return_tensors="np")
        pixel_values = enc["pixel_values"].astype(np.float32)
        out = self.vision_session.run(None, {self.vision_input_name: pixel_values})[0]
        return self._normalize(out)

    def embed_text(self, text: str) -> np.ndarray:
        enc = self.processor(text=[text], return_tensors="np", truncation=True, padding=True)

        input_feed = {}
        for name in self.text_input_names:
            if name in enc:
                input_feed[name] = enc[name].astype(np.int64)

        if "attention_mask" in self.text_input_names and "attention_mask" not in input_feed:
            input_feed["attention_mask"] = np.ones_like(enc["input_ids"], dtype=np.int64)

        out = self.text_session.run(None, input_feed)[0]
        return self._normalize(out)

    def compose_query(self, image_emb: np.ndarray, text_emb: np.ndarray, alpha: float) -> np.ndarray:
        q = alpha * image_emb + (1.0 - alpha) * text_emb
        return self._normalize(q)
