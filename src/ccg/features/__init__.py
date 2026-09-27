"""Feature extraction layer (placeholder until the data/weights download).

The package must stay importable without torch, open_clip, GPU or weights: all
heavy imports live inside functions (see :mod:`ccg.features.clip_encoder`).
The three ``extract_*`` modules are Phase 0 CLI stubs and intentionally raise
``NotImplementedError``.
"""

from __future__ import annotations

from .clip_encoder import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
    FEATURE_DIM,
    ClipEncoder,
    ClipEncoderConfig,
    build_encoder,
    l2_normalize,
)

__all__ = [
    "FEATURE_DIM",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_PRETRAINED",
    "ClipEncoder",
    "ClipEncoderConfig",
    "build_encoder",
    "l2_normalize",
]
