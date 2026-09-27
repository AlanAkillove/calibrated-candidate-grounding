"""Frozen OpenCLIP ViT-B/32 wrapper (placeholder, GPU/weights optional).

Everything heavy is imported *inside* functions, so importing this module never
requires torch, open_clip or a downloaded checkpoint - the whole package must
stay importable on a machine that has neither GPU nor weights (Phase 0 order:
audit data first, extract features later).

No network access happens implicitly: :func:`build_encoder` refuses to download
unless the caller passes ``allow_download=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

__all__ = [
    "FEATURE_DIM",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_PRETRAINED",
    "ClipEncoderConfig",
    "ClipEncoder",
    "build_encoder",
    "l2_normalize",
]

#: ViT-B/32 (laion2b_s34b_b79k) joint embedding dimensionality (section 3).
FEATURE_DIM = 512
DEFAULT_MODEL_NAME = "ViT-B/32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


def l2_normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalisation (numpy, used by every cached-feature path)."""
    arr = np.asarray(vectors, dtype=np.float32)
    single = arr.ndim == 1
    matrix = arr.reshape(1, -1) if single else arr
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    out = matrix / np.maximum(norms, eps)
    return out.reshape(-1).astype(np.float32) if single else out.astype(np.float32, copy=False)


@dataclass
class ClipEncoderConfig:
    """What is needed to rebuild the *same* frozen encoder later (audit trail)."""

    model_name: str = DEFAULT_MODEL_NAME
    pretrained: str = DEFAULT_PRETRAINED
    embed_dim: int = FEATURE_DIM
    device: str = "cpu"
    precision: str = "fp16"
    context_pad: int = 8
    checkpoint_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "backbone": f"open_clip {self.model_name} {self.pretrained}",
            "embed_dim": int(self.embed_dim),
            "device": str(self.device),
            "precision": str(self.precision),
            "context_pad": int(self.context_pad),
            "checkpoint_path": self.checkpoint_path,
        }


class ClipEncoder:
    """Thin wrapper around a frozen ``open_clip`` model.

    Only the *interface* is guaranteed here; the model itself is created by
    :func:`build_encoder`.  Feature extraction is deliberately exposed as three
    methods matching the cache layout of protocol section 7 (global image
    features, region-crop features, query text features).
    """

    def __init__(self, model, preprocess, tokenize, config: ClipEncoderConfig) -> None:
        self.model = model
        self.preprocess = preprocess
        self.tokenize = tokenize
        self.config = config

    # -- encoding ------------------------------------------------------------
    def encode_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """``[M, D]`` L2-normalised text embeddings for the referring expressions."""
        torch = self._torch()
        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), int(batch_size)):
                batch = list(texts[start : start + batch_size])
                tokens = self.tokenize(batch).to(self._device(torch))
                if self.config.precision == "fp16" and self._device(torch).type == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        features = self.model.encode_text(tokens)
                else:
                    features = self.model.encode_text(tokens)
                chunks.append(features.float().cpu().numpy())
        return l2_normalize(np.concatenate(chunks, axis=0))

    def encode_image_crops(
        self, images: Iterable, boxes: Optional[np.ndarray] = None, batch_size: int = 64
    ) -> np.ndarray:
        """``[N, D]`` embeddings of PIL crops (region features, section 7)."""
        raise NotImplementedError(
            "Phase 0 pipeline: enabled after the data download (scripts/extract_features.py)"
        )

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _torch():
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("ClipEncoder requires torch") from exc
        return torch

    def _device(self, torch):
        return torch.device(self.config.device)


def build_encoder(
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: str = DEFAULT_PRETRAINED,
    *,
    device: str = "cpu",
    precision: str = "fp16",
    checkpoint_path: Optional[str | Path] = None,
    allow_download: bool = False,
) -> ClipEncoder:
    """Create the frozen OpenCLIP encoder (skeleton - requires open_clip + weights).

    ``open_clip`` is imported here, not at module import time.  With
    ``allow_download=False`` (default) a local ``checkpoint_path`` is mandatory:
    this keeps unit tests and code review offline-only, as the protocol demands.
    """
    if not allow_download and checkpoint_path is None:
        raise RuntimeError(
            "build_encoder() will not download weights implicitly. Pass checkpoint_path=... "
            "to a locally cached open_clip checkpoint, or allow_download=True once the "
            "weights have been fetched deliberately."
        )
    try:
        import open_clip  # noqa: F401  (local import: heavy, optional dependency)
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "open_clip_torch / torch are required to build the encoder. They are declared in "
            "pyproject.toml but not needed for the CPU-only Phase 0 unit tests."
        ) from exc

    config = ClipEncoderConfig(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        precision=precision,
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=None if config.checkpoint_path is None else pretrained,
        pretrained_path=config.checkpoint_path,
    )
    model = model.to(torch.device(device)).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenize = open_clip.get_tokenizer(model_name)
    return ClipEncoder(model=model, preprocess=preprocess, tokenize=tokenize, config=config)
