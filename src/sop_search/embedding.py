from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str
    kind: str


MODELS = {
    "clip-vit-b32": ModelSpec(
        "openai/clip-vit-base-patch32", "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268", "clip"
    ),
    "dinov2-small": ModelSpec(
        "facebook/dinov2-small", "ed25f3a31f01632728cabb09d1542f84ab7b0056", "dinov2"
    ),
}


def load_image(path: Path | str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


class Embedder:
    def __init__(self, name: str):
        self.name = name
        self.spec = MODELS[name]
        self.processor = AutoImageProcessor.from_pretrained(
            self.spec.repo_id, revision=self.spec.revision
        )
        if self.spec.kind == "clip":
            self.model = CLIPVisionModelWithProjection.from_pretrained(
                self.spec.repo_id, revision=self.spec.revision
            )
            self.dim = self.model.config.projection_dim
        else:
            self.model = AutoModel.from_pretrained(self.spec.repo_id, revision=self.spec.revision)
            self.dim = self.model.config.hidden_size
        self.model.eval()

    def preprocess(self, images: Sequence[Image.Image]) -> torch.Tensor:
        return self.processor(images=list(images), return_tensors="pt")["pixel_values"]

    @torch.inference_mode()
    def embed_pixels(self, pixel_values: torch.Tensor) -> np.ndarray:
        out = self.model(pixel_values=pixel_values)
        feats = out.image_embeds if self.spec.kind == "clip" else out.pooler_output
        return l2_normalize(feats.float().cpu().numpy())

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        return self.embed_pixels(self.preprocess(images))

    def embed_paths(self, paths: Sequence[Path], batch_size: int = 32) -> np.ndarray:
        chunks = []
        for start in range(0, len(paths), batch_size):
            batch = [load_image(p) for p in paths[start : start + batch_size]]
            chunks.append(self.embed_images(batch))
        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate(chunks, axis=0)
