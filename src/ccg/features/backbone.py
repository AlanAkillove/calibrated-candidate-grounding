"""Unified frozen-backbone factory for the V2-G generalization round (V2-A1).

V1 hard-wired the whole feature pipeline to the OpenCLIP ViT-B/32 encoder in
:mod:`ccg.features.clip_encoder`.  V2-G asks whether the two headline findings
survive *other* frozen vision-language backbones, so this module adds a second
backend - **SigLIP base-patch16-224** through ``transformers`` - behind the very
same duck-typed encoder interface the extraction scripts already consume:

    ``encode_images`` / ``encode_texts`` / ``encode_crops`` -> float16, **always
    L2-normalised** (the V2-A1 amendment: every V2 cache stores cosine-ready
    rows regardless of native contrastive objective);
    ``feature_dim`` / ``context_length`` / ``logit_scale`` / ``token_counts``;
    a ``config`` carrying the audit-trail fields the CLIs log, and ``metadata()``.

Two honest differences from CLIP are recorded rather than hidden:

* **objective** - SigLIP trains a *sigmoid* pairwise loss, so it exposes a
  ``logit_bias`` as well as ``logit_scale``; the CLIP-family backbones use a
  softmax *cross-entropy* over the batch.  B3 is candidate-blind and consumes
  only the L2-normalised embeddings, so the objective enters the audit only.
* **tokeniser** - SigLIP uses a GPT2-style SentencePiece tokenizer whose
  ``model_max_length`` is 64, *shorter* than CLIP's 77.  It does **not**
  truncate by default, so ``encode_texts`` passes ``truncation=True`` explicitly
  and the extraction text path counts truncated sentences as provenance.

No network is ever required: both backbones load from an explicit local path
(``checkpoint_path``).  For SigLIP that path is the *directory* holding
``config.json`` + ``model.safetensors`` + the tokenizer files.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .clip_encoder import (
    CROP_POLICY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
    DEFAULT_RESOLUTION,
    crop_image_at_boxes,
    l2_normalize,
    load_rgb_image,
    resolve_hf_endpoint,
    _pil,
)
from .clip_encoder import build_encoder as _build_open_clip_encoder

__all__ = [
    "BACKEND_OPEN_CLIP",
    "BACKEND_SIGLIP",
    "SIGLIP_DEFAULT_REPO_ID",
    "SIGLIP_DEFAULT_LOCAL_DIR",
    "SiglipEncoderConfig",
    "SiglipEncoder",
    "build_siglip_encoder",
    "build_encoder",
]

#: Backend tags understood by :func:`build_encoder`.
BACKEND_OPEN_CLIP = "open_clip"
BACKEND_SIGLIP = "siglip"

#: HuggingFace repo id of the SigLIP base backbone (recorded for provenance only;
#: the weights are loaded from ``SIGLIP_DEFAULT_LOCAL_DIR`` without a network call).
SIGLIP_DEFAULT_REPO_ID = "google/siglip-base-patch16-224"
#: Where the audited SigLIP files live on the offline target machine.
SIGLIP_DEFAULT_LOCAL_DIR = "cache/hf_local/siglip_b16_224"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class SiglipEncoderConfig:
    """Rebuild-the-same-frozen-encoder record (mirrors :class:`ClipEncoderConfig`)."""

    model_name: str = SIGLIP_DEFAULT_REPO_ID
    pretrained: str = SIGLIP_DEFAULT_REPO_ID
    device: str = "cpu"
    precision: str = "fp16"
    batch_size: int = DEFAULT_BATCH_SIZE
    resolution: int = DEFAULT_RESOLUTION
    library_version: str = ""
    checkpoint_path: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    hf_endpoint: Optional[str] = None
    hf_endpoint_source: Optional[str] = None
    tokenizer_name: str = ""
    tokenizer_context_length: int = 64
    objective: str = "sigmoid"
    feature_dim: int = 768
    preprocessing: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------


class SiglipEncoder:
    """Thin wrapper around a frozen ``transformers`` SigLIP model (eval, no grads).

    Same public surface as :class:`ccg.features.clip_encoder.ClipEncoder` so the
    extraction scripts run unchanged against either backend; only the internals
    (image processor, tokenizer, ``get_*_features``) differ.
    """

    def __init__(self, model, processor, config: SiglipEncoderConfig) -> None:
        self.model = model
        self.processor = processor
        self.image_processor = processor.image_processor
        self.tokenizer = processor.tokenizer
        self.config = config
        self._device = None
        # Public callable the text extraction path may wrap for its progress bar;
        # ``encode_texts`` always routes through it (list[str] -> BatchEncoding).
        self.tokenize = self._tokenize

    # -- basics ------------------------------------------------------------
    @staticmethod
    def _torch():
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("SiglipEncoder requires torch") from exc
        return torch

    @property
    def torch_device(self):
        torch = self._torch()
        if self._device is None:
            self._device = torch.device(self.config.device)
        return self._device

    @property
    def precision(self) -> str:
        return self.config.precision

    @property
    def feature_dim(self) -> int:
        return int(self.config.feature_dim)

    @property
    def context_length(self) -> int:
        return int(self.config.tokenizer_context_length)

    @property
    def logit_scale(self) -> float:
        """``model.logit_scale.exp()`` - the native SigLIP temperature."""
        return float(self.model.logit_scale.exp().item())

    @property
    def logit_bias(self) -> float:
        """SigLIP's learned bias term (absent for a CLIP-style backbone)."""
        bias = getattr(self.model, "logit_bias", None)
        return float(bias.item()) if bias is not None else float("nan")

    def _autocast(self, torch):
        if self.precision == "fp16" and self.torch_device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    # -- tokenisation (list[str] -> padded/truncated batch) ----------------
    def _tokenize(self, texts: Sequence[str]):
        return self.tokenizer(
            [str(t) for t in texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.context_length,
        )

    # -- encoding ----------------------------------------------------------
    def encode_images(self, images: Iterable, batch_size: Optional[int] = None) -> np.ndarray:
        """``[M, feature_dim]`` float16, L2-normalised, for a list of PIL images."""
        torch = self._torch()
        pil_images = list(images)
        if not pil_images:
            return np.zeros((0, self.feature_dim), dtype=np.float16)
        bs = int(batch_size or self.config.batch_size)
        if bs <= 0:
            raise ValueError(f"batch_size must be positive, got {bs}")
        device = self.torch_device
        self.model.eval()
        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(pil_images), bs):
                batch = [load_rgb_image_bytes_or_pil(img) for img in pil_images[start : start + bs]]
                # Force channels-last so the slow SigLIP processor never mis-infers a
                # degenerate crop (a proposal with width or height == 1, or a 3xW thin
                # crop whose numpy array is (3, W, 3)) as channels-first, which raises a
                # PIL "Cannot handle this data type" TypeError.  All real crops are HWC,
                # so this is behaviour-preserving for every unambiguous image.
                pixels = self.image_processor(
                    images=batch, return_tensors="pt", input_data_format="channels_last"
                )["pixel_values"]
                pixels = pixels.to(device)
                with self._autocast(torch):
                    features = self.model.get_image_features(pixel_values=pixels)
                chunks.append(np.asarray(features.float().cpu().numpy()))
        normalized = l2_normalize(np.concatenate(chunks, axis=0))
        return normalized.astype(np.float16)

    def encode_crops(
        self, image, boxes, batch_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Frozen crop policy + L2-normalised crop rows (invalid boxes -> zero rows)."""
        boxes_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self._torch()
        crops, valid, _ = crop_image_at_boxes(image, boxes_arr)
        features = np.zeros((boxes_arr.shape[0], self.feature_dim), dtype=np.float16)
        if valid.any():
            features[valid] = self.encode_images(crops, batch_size=batch_size)
        return features, valid

    def encode_texts(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        """``[M, feature_dim]`` float16, L2-normalised, for a list of raw strings."""
        torch = self._torch()
        texts = [str(t) for t in texts]
        if not texts:
            return np.zeros((0, self.feature_dim), dtype=np.float16)
        bs = int(batch_size or self.config.batch_size)
        if bs <= 0:
            raise ValueError(f"batch_size must be positive, got {bs}")
        device = self.torch_device
        self.model.eval()
        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), bs):
                batch = self.tokenize(texts[start : start + bs])
                batch = {k: v.to(device) for k, v in batch.items()}
                with self._autocast(torch):
                    features = self.model.get_text_features(**batch)
                chunks.append(np.asarray(features.float().cpu().numpy()))
        normalized = l2_normalize(np.concatenate(chunks, axis=0))
        return normalized.astype(np.float16)

    def token_counts(self, texts: Sequence[str]) -> np.ndarray:
        """SentencePiece token counts (no added special tokens for SigLIP).

        ``_tokenize`` truncates at ``context_length`` (64), so any value above
        that means the stored embedding was computed on a truncated text - the
        extraction path counts these as provenance.
        """
        counts: List[int] = []
        for text in texts:
            ids = self.tokenizer([str(text)])["input_ids"][0]
            counts.append(len(ids))
        return np.asarray(counts, dtype=np.int64)

    # -- provenance --------------------------------------------------------
    def metadata(self) -> dict:
        """The frozen ``metadata.json["backbone"]`` record (V2-A1 layout)."""
        return {
            "library": "transformers",
            "library_version": self.config.library_version,
            "backend": BACKEND_SIGLIP,
            "model_name": self.config.model_name,
            "pretrained": self.config.pretrained,
            "checkpoint_path": self.config.checkpoint_path,
            "checkpoint_sha256": self.config.checkpoint_sha256,
            "embedding_dim": self.feature_dim,
            "objective": self.config.objective,
            "precision": self.config.precision,
            "resolution": int(self.config.resolution),
            "preprocessing": dict(self.config.preprocessing),
            "native_logit_scale": self.logit_scale,
            "native_logit_bias": self.logit_bias,
            "l2_normalized": True,
            "tokenizer": {
                "name": self.config.tokenizer_name,
                "context_length": int(self.config.tokenizer_context_length),
            },
        }


def load_rgb_image_bytes_or_pil(image):
    """Accept a PIL image (pass-through) or a path (load as RGB)."""
    Image = _pil()
    if isinstance(image, Image.Image):
        return image if image.mode == "RGB" else image.convert("RGB")
    return load_rgb_image(image)


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def _siglip_preprocessing(image_processor, config) -> dict:
    """Read the *actual* SigLIP preprocessing (fixed resize, mean/std 0.5)."""
    size = getattr(image_processor, "size", None) or {}
    if isinstance(size, dict):
        height = int(size.get("height", DEFAULT_RESOLUTION))
        width = int(size.get("width", DEFAULT_RESOLUTION))
    else:  # pragma: no cover - defensive
        height = width = int(size)
    return {
        "resize_mode": "square (fixed resize, no center crop)",
        "resize_size": [height, width],
        "interpolation": _resample_name(getattr(image_processor, "resample", None)),
        "center_crop": None,
        "mean": [float(v) for v in np.ravel(getattr(image_processor, "image_mean", []))],
        "std": [float(v) for v in np.ravel(getattr(image_processor, "image_std", []))],
        "rescale_factor": float(getattr(image_processor, "rescale_factor", 1.0 / 255)),
        "crop_policy": CROP_POLICY,
        "min_crop_px": 1,
        "source": "introspected from the live SiglipImageProcessor",
    }


def _resample_name(resample) -> str:
    mapping = {0: "nearest", 1: "lanez", 2: "bilinear", 3: "bicubic", 4: "lanczos"}
    if isinstance(resample, int):
        return mapping.get(resample, str(resample))
    return str(getattr(resample, "name", resample)).lower()


def _probe_siglip_dim(model, image_processor, device, torch, fallback: int) -> int:
    """One dummy image through ``get_image_features`` to read the true width."""
    try:
        Image = _pil()
        dummy = Image.new("RGB", (32, 32))
        pixels = image_processor(images=[dummy], return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            out = model.get_image_features(pixel_values=pixels)
        dim = int(out.shape[-1])
        if dim > 0:
            return dim
    except Exception:  # pragma: no cover - defensive
        pass
    return int(fallback)


def build_siglip_encoder(
    device: str = "cuda",
    precision: str = "fp16",
    model_path: Optional[str | Path] = None,
    *,
    repo_id: str = SIGLIP_DEFAULT_REPO_ID,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hf_endpoint: Optional[str] = None,
    compute_sha256: bool = False,
) -> SiglipEncoder:
    """Build the frozen SigLIP base-patch16-224 encoder from a local directory.

    ``model_path`` is the directory holding ``config.json`` + ``model.safetensors``
    + the tokenizer files; it defaults to :data:`SIGLIP_DEFAULT_LOCAL_DIR`.  The
    model is loaded with ``local_files_only=True`` so the offline machine never
    attempts a download.
    """
    if precision not in ("fp16", "fp32"):
        raise ValueError(f"precision must be 'fp16' or 'fp32', got {precision!r}")
    endpoint, endpoint_source = resolve_hf_endpoint(hf_endpoint)

    try:
        import torch
        from transformers import AutoProcessor, SiglipModel
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "transformers / torch / sentencepiece are required for the SigLIP backend"
        ) from exc

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "build_siglip_encoder(device='cuda') requested but CUDA is not available; "
            "no silent CPU fallback (feature extraction targets the frozen GPU path)"
        )

    src = str(Path(model_path or SIGLIP_DEFAULT_LOCAL_DIR))
    if not Path(src).exists():
        raise FileNotFoundError(
            f"SigLIP weights directory {src} does not exist; download the repo files "
            "into it (config.json + model.safetensors + tokenizer + spiece.model) or "
            "pass model_path=..."
        )

    processor = AutoProcessor.from_pretrained(src, local_files_only=True)
    model = SiglipModel.from_pretrained(src, local_files_only=True)
    model = model.to(torch_device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    feature_dim = _probe_siglip_dim(model, processor.image_processor, torch_device, torch, 768)

    checkpoint_sha256 = None
    if compute_sha256:  # pragma: no cover - off by default (large file)
        from ..utils.io import file_sha256

        weights = sorted(Path(src).glob("model.safetensors")) + sorted(
            Path(src).glob("pytorch_model.bin")
        )
        if weights:
            checkpoint_sha256 = file_sha256(weights[0])

    config = SiglipEncoderConfig(
        model_name=repo_id,
        pretrained=repo_id,
        device=str(torch_device),
        precision=precision,
        batch_size=int(batch_size),
        resolution=DEFAULT_RESOLUTION,
        library_version=str(_transformers_version()),
        checkpoint_path=src,
        checkpoint_sha256=checkpoint_sha256,
        hf_endpoint=endpoint,
        hf_endpoint_source=endpoint_source,
        tokenizer_name=type(processor.tokenizer).__name__,
        tokenizer_context_length=int(
            getattr(processor.tokenizer, "model_max_length", 64) or 64
        ),
        feature_dim=int(feature_dim),
        preprocessing=_siglip_preprocessing(processor.image_processor, model.config),
    )
    return SiglipEncoder(model=model, processor=processor, config=config)


def _transformers_version():  # pragma: no cover - trivial
    try:
        import transformers

        return getattr(transformers, "__version__", "unknown")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# unified factory
# ---------------------------------------------------------------------------


def build_encoder(
    device: str = "cuda",
    precision: str = "fp16",
    cache_dir: Optional[str | Path] = None,
    *,
    backend: str = BACKEND_OPEN_CLIP,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: str = DEFAULT_PRETRAINED,
    checkpoint_path: Optional[str | Path] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hf_endpoint: Optional[str] = None,
    download_attempts: int = 10,
    download_retry_delay: float = 3.0,
    compute_sha256: bool = True,
):
    """Build a frozen encoder for the requested ``backend``.

    ``open_clip`` (the V1 default) dispatches to
    :func:`ccg.features.clip_encoder.build_encoder`; ``siglip`` dispatches to
    :func:`build_siglip_encoder`, where ``checkpoint_path`` is the *directory*
    of the SigLIP files.  Both return an object with the same public surface
    (``encode_images`` / ``encode_texts`` / ``encode_crops`` / ``feature_dim`` /
    ``metadata()`` / ``logit_scale`` / ``token_counts`` / ``config``).
    """
    if backend == BACKEND_OPEN_CLIP:
        return _build_open_clip_encoder(
            device=device,
            precision=precision,
            cache_dir=cache_dir,
            model_name=model_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            hf_endpoint=hf_endpoint,
            download_attempts=download_attempts,
            download_retry_delay=download_retry_delay,
            compute_sha256=compute_sha256,
        )
    if backend == BACKEND_SIGLIP:
        return build_siglip_encoder(
            device=device,
            precision=precision,
            model_path=checkpoint_path if checkpoint_path is not None else cache_dir,
            repo_id=pretrained or SIGLIP_DEFAULT_REPO_ID,
            batch_size=batch_size,
            hf_endpoint=hf_endpoint,
            compute_sha256=compute_sha256,
        )
    raise ValueError(
        f"unknown backend {backend!r}; expected {BACKEND_OPEN_CLIP!r} or {BACKEND_SIGLIP!r}"
    )
