"""Frozen OpenCLIP ViT-B/32 encoder - real implementation (protocol §4/§7).

Heavy imports (``torch``, ``open_clip``, ``PIL``, ``torchvision``) all happen
inside functions, so importing this module stays cheap (numpy + stdlib only)
and the pure geometry helpers below are testable without a GPU.

Frozen crop policy (multi-agent brief §6/§7, 2026-09-27): **exact proposal
crop** - an xyxy float32 box is rounded to integer pixels (``np.rint``,
round-half-to-even), clamped to the image rectangle, and if the resulting crop
is smaller than 1 px in either dimension the proposal is flagged *invalid*.
Nothing is silently expanded and nothing is silently dropped.  Valid crops go
through the exact ``create_model_and_transforms`` **val** preprocessing
(resize shortest edge -> 224 bicubic, center-crop 224, CLIP mean/std).

HuggingFace mirror behaviour (recorded): direct ``huggingface.co`` downloads
fail on the target machine, but ``hf-mirror.com`` works.  :func:`build_encoder`
therefore calls :func:`resolve_hf_endpoint`, which

1. honours an explicit ``hf_endpoint`` argument, else
2. keeps an already-exported ``HF_ENDPOINT`` environment variable, else
3. applies ``os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")``

and patches ``huggingface_hub.constants.ENDPOINT`` when the library was
imported earlier in the process (the mirror default must not be silently
skipped because another module imported huggingface_hub first).  The returned
``(endpoint, source)`` pair is stored on the encoder config and recorded in
``extraction_stats``.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.io import file_sha256

__all__ = [
    "FEATURE_DIM",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_PRETRAINED",
    "DEFAULT_RESOLUTION",
    "DEFAULT_BATCH_SIZE",
    "MIN_CROP_PX",
    "CROP_POLICY",
    "HF_MIRROR_ENDPOINT",
    "CLIP_MEAN",
    "CLIP_STD",
    "ClipEncoderConfig",
    "ClipEncoder",
    "build_encoder",
    "l2_normalize",
    "resolve_hf_endpoint",
    "proposal_crop_boxes",
    "crop_image_at_boxes",
    "describe_preprocessing",
    "load_rgb_image",
]

#: Embedding width of the *default* (V1) OpenCLIP ViT-B/32 backbone.  Kept as a
#: module constant for backward-compatible call sites; the authoritative width
#: of a built encoder is ``ClipEncoder.feature_dim`` (probed from the live model
#: so ViT-B/16 and any other open_clip checkpoint report its true dimensionality).
FEATURE_DIM = 512
DEFAULT_MODEL_NAME = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"
DEFAULT_RESOLUTION = 224
DEFAULT_BATCH_SIZE = 128

#: Frozen crop policy: crops thinner than this (either axis, integer px) are invalid.
MIN_CROP_PX = 1
#: Human-readable crop policy string recorded in every metadata.json.
CROP_POLICY = "exact proposal crop (round->clamp), no context padding"

#: Reachable HuggingFace mirror when huggingface.co is blocked (recorded behaviour).
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
#: Filename open_clip uses inside the HF repo of a pretrained checkpoint.
HF_CHECKPOINT_FILENAME = "open_clip_pytorch_model.bin"

#: OpenAI CLIP preprocessing constants (also the open_clip pretrained-cfg values).
CLIP_MEAN: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711)


# ---------------------------------------------------------------------------
# numpy helpers (no heavy imports)
# ---------------------------------------------------------------------------


def l2_normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise L2 normalisation in float32 (single vectors stay 1-d)."""
    arr = np.asarray(vectors, dtype=np.float32)
    single = arr.ndim == 1
    matrix = arr.reshape(1, -1) if single else arr
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    out = matrix / np.maximum(norms, eps)
    return out.reshape(-1) if single else out


def resolve_hf_endpoint(endpoint: Optional[str] = None) -> Tuple[str, str]:
    """Return ``(effective_endpoint, source)`` and make huggingface_hub use it.

    See the module docstring: the default mirror is applied with
    ``os.environ.setdefault`` unless the caller or the environment already
    chose an endpoint.  When ``huggingface_hub`` was imported earlier (so its
    ``ENDPOINT`` constant was frozen before this call), the constant is
    patched to the effective endpoint - otherwise the mirror would be
    silently ignored.
    """
    if endpoint:
        os.environ["HF_ENDPOINT"] = str(endpoint)
        source = "explicit-argument"
    elif os.environ.get("HF_ENDPOINT"):
        source = "environment"
    else:
        os.environ.setdefault("HF_ENDPOINT", HF_MIRROR_ENDPOINT)
        source = "hf-mirror-default"
    effective = os.environ["HF_ENDPOINT"]
    try:
        import huggingface_hub.constants as constants  # local import: optional dep

        if getattr(constants, "ENDPOINT", None) != effective:
            constants.ENDPOINT = effective
    except Exception:  # pragma: no cover - huggingface_hub absent
        pass
    return effective, source


def _pil():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("Pillow is required to load/crop images") from exc
    return Image


def load_rgb_image(path: str | Path):
    """Load an image from disk as PIL ``RGB`` (the only image loader used)."""
    Image = _pil()
    with Image.open(path) as handle:
        return handle.convert("RGB")


def proposal_crop_boxes(
    boxes, image_size: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Frozen crop policy, geometry only (no pixels touched).

    Parameters
    ----------
    boxes:
        ``[M, 4]`` xyxy boxes in original-image pixels (float32 from the bank).
    image_size:
        ``(width, height)`` of the image (PIL convention).

    Returns
    -------
    int_boxes : ``[M, 4]`` int64
        xyxy after ``np.rint`` (round-half-to-even) and clamping to the image;
    valid : ``[M]`` bool
        ``False`` where the clamped crop is < :data:`MIN_CROP_PX` in either
        dimension (a degenerate/off-image proposal is never expanded).
    """
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.int64), np.zeros((0,), dtype=bool)
    arr = arr.reshape(-1, 4)
    if not np.isfinite(arr).all():
        raise ValueError("boxes contain non-finite values; refusing to crop")
    width, height = int(image_size[0]), int(image_size[1])
    rounded = np.rint(arr).astype(np.int64)
    rounded[:, [0, 2]] = np.clip(rounded[:, [0, 2]], 0, width)
    rounded[:, [1, 3]] = np.clip(rounded[:, [1, 3]], 0, height)
    valid = (rounded[:, 2] - rounded[:, 0] >= MIN_CROP_PX) & (
        rounded[:, 3] - rounded[:, 1] >= MIN_CROP_PX
    )
    return rounded, valid


def crop_image_at_boxes(image, boxes):
    """Crop a PIL image at every valid box; degenerate boxes yield no crop.

    Returns ``(crops, valid_mask, int_boxes)`` where ``crops`` holds one PIL
    crop per valid box (in box order, never padded, never expanded).
    """
    Image = _pil()
    if not isinstance(image, Image.Image):
        raise TypeError(f"expected a PIL.Image, got {type(image).__name__}")
    image = image if image.mode == "RGB" else image.convert("RGB")
    int_boxes, valid = proposal_crop_boxes(boxes, image.size)
    crops = [image.crop(tuple(int(v) for v in row)) for row in int_boxes[valid]]
    return crops, valid, int_boxes


def describe_preprocessing(transform) -> dict:
    """Record the *actual* preprocessing chain of the live val transform.

    Reads the Resize / CenterCrop / Normalize operators of the compiled
    ``open_clip`` val transform; falls back to the documented open_clip
    constants (and says so in ``"source"``) if introspection is impossible.
    """
    info: dict = {
        "resize_mode": "shortest",
        "resize_size": DEFAULT_RESOLUTION,
        "interpolation": "bicubic",
        "center_crop": DEFAULT_RESOLUTION,
        "mean": list(CLIP_MEAN),
        "std": list(CLIP_STD),
        "crop_policy": CROP_POLICY,
        "min_crop_px": MIN_CROP_PX,
        "source": "open_clip ViT-B/32 fallback constants (transform introspection failed)",
    }
    chain = getattr(transform, "transforms", None)
    if not chain:
        return info
    try:
        from torchvision import transforms as T

        for op in chain:
            if isinstance(op, T.Resize):
                size = op.size
                info["resize_size"] = (
                    int(size) if isinstance(size, int) else [int(s) for s in size]
                )
                interpolation = getattr(op, "interpolation", None)
                if interpolation is not None:
                    info["interpolation"] = str(
                        getattr(interpolation, "name", interpolation)
                    ).lower()
                antialias = getattr(op, "antialias", None)
                if antialias is not None:
                    info["antialias"] = bool(antialias)
            elif isinstance(op, T.CenterCrop):
                size = op.size
                info["center_crop"] = int(size) if isinstance(size, int) else [int(s) for s in size]
            elif isinstance(op, T.Normalize):
                info["mean"] = [float(v) for v in np.ravel(op.mean)]
                info["std"] = [float(v) for v in np.ravel(op.std)]
        info["source"] = "introspected from the live open_clip val transform"
    except Exception as exc:  # pragma: no cover - defensive
        info["introspection_error"] = f"{type(exc).__name__}: {exc}"
    return info


# ---------------------------------------------------------------------------
# checkpoint location / download
# ---------------------------------------------------------------------------


def _hf_coordinates(model_name: str, pretrained: str) -> Tuple[str, str]:
    """``(repo_id, filename)`` of the official open_clip checkpoint on the Hub."""
    try:
        from open_clip.pretrained import get_pretrained_cfg
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("open_clip_torch is required to locate the checkpoint") from exc
    cfg = get_pretrained_cfg(model_name, pretrained)
    repo_id = str(cfg.get("hf_hub") or "").strip("/")
    if not repo_id:
        raise RuntimeError(
            f"no HuggingFace coordinates for open_clip {model_name} / {pretrained}"
        )
    return repo_id, HF_CHECKPOINT_FILENAME


def _find_cached_checkpoint(
    repo_id: str, filename: str, cache_dir: Optional[str | Path]
) -> Optional[Path]:
    """Locate an already-downloaded checkpoint without any network call."""
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        if isinstance(hit, str) and Path(hit).exists():
            return Path(hit)
    except Exception:  # pragma: no cover - defensive
        pass
    if cache_dir is not None:
        base = Path(cache_dir)
    else:
        hf_home = os.environ.get("HF_HOME")
        base = (
            Path(hf_home) / "hub"
            if hf_home
            else Path.home() / ".cache" / "huggingface" / "hub"
        )
    repo_dir = base / ("models--" + repo_id.replace("/", "--"))
    matches = sorted(repo_dir.glob(f"snapshots/*/{filename}"))
    return matches[0] if matches else None


def _ensure_checkpoint(
    model_name: str,
    pretrained: str,
    cache_dir: Optional[str | Path],
    checkpoint_path: Optional[str | Path],
    *,
    attempts: int = 10,
    retry_delay: float = 3.0,
) -> Path:
    """Return the local checkpoint file, downloading it with bounded retries.

    ``hf_hub_download`` resumes partial downloads (``.incomplete`` files), so
    retrying recovers from the flaky TLS resets observed on the mirror
    network.  An explicit ``checkpoint_path`` skips the download entirely.
    """
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint_path {path} does not exist")
        return path
    repo_id, filename = _hf_coordinates(model_name, pretrained)
    cached = _find_cached_checkpoint(repo_id, filename, cache_dir)
    if cached is not None:
        return cached
    from huggingface_hub import hf_hub_download

    last_error: Optional[BaseException] = None
    for attempt in range(1, int(attempts) + 1):
        try:
            path = Path(hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir))
            return path
        except Exception as exc:  # noqa: BLE001 - retried below with backoff
            last_error = exc
            if attempt < attempts:
                time.sleep(min(float(retry_delay) * attempt, 30.0))
    raise RuntimeError(
        f"failed to download {repo_id}/{filename} after {attempts} attempts; "
        f"last error: {last_error!r}. The default endpoint is {HF_MIRROR_ENDPOINT} "
        "(override with HF_ENDPOINT); pass checkpoint_path=... to skip the download."
    )


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------


@dataclass
class ClipEncoderConfig:
    """Everything needed to rebuild the *same* frozen encoder later (audit trail)."""

    model_name: str = DEFAULT_MODEL_NAME
    pretrained: str = DEFAULT_PRETRAINED
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
    tokenizer_context_length: int = 77
    #: joint embedding width of the built backbone (512 for ViT-B/32 and ViT-B/16,
    #: probed from the model when a different open_clip checkpoint is loaded).
    feature_dim: int = FEATURE_DIM
    preprocessing: dict = field(default_factory=dict)


class ClipEncoder:
    """Thin wrapper around a frozen ``open_clip`` model (eval, no gradients).

    * :meth:`encode_images` - L2-normalised embeddings of whole PIL images;
    * :meth:`encode_crops` - frozen crop policy + L2-normalised crop rows,
      plus the ``valid_mask`` (invalid crops get an all-zero row);
    * :meth:`encode_texts` - L2-normalised query embeddings;
    * :meth:`metadata` / :attr:`logit_scale` - the provenance record.
    """

    def __init__(self, model, preprocess, tokenize, config: ClipEncoderConfig) -> None:
        self.model = model
        self.preprocess = preprocess
        self.tokenize = tokenize
        self.config = config
        self._device = None

    # -- basics ------------------------------------------------------------
    @staticmethod
    def _torch():
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("ClipEncoder requires torch") from exc
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
        """Joint embedding width of *this* backbone (probed at build time)."""
        return int(self.config.feature_dim)

    @property
    def logit_scale(self) -> float:
        """``model.logit_scale.exp()`` - the native CLIP temperature (1/scale)."""
        return float(self.model.logit_scale.exp().item())

    @property
    def context_length(self) -> int:
        return int(self.config.tokenizer_context_length)

    def _autocast(self, torch):
        if self.precision == "fp16" and self.torch_device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    # -- encoding ----------------------------------------------------------
    def encode_images(self, images: Iterable, batch_size: Optional[int] = None) -> np.ndarray:
        """``[M, 512]`` float16, L2-normalised, for a list of PIL images."""
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
                batch = pil_images[start : start + bs]
                tensors = torch.stack([self.preprocess(_as_rgb(img)) for img in batch])
                tensors = tensors.to(device)
                with self._autocast(torch):
                    features = self.model.encode_image(tensors)
                chunks.append(np.asarray(features.float().cpu().numpy()))
        normalized = l2_normalize(np.concatenate(chunks, axis=0))
        return normalized.astype(np.float16)

    def encode_crops(
        self, image, boxes, batch_size: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode every proposal crop of one image under the frozen crop policy.

        Returns ``(features [M, 512] float16, valid_mask [M] bool)``; rows of
        invalid boxes stay all-zero (they are still part of the cache layout,
        the mask is the authoritative record of what was encoded).
        """
        boxes_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self._torch()  # fail early without torch, before touching pixels
        crops, valid, _ = crop_image_at_boxes(image, boxes_arr)
        features = np.zeros((boxes_arr.shape[0], self.feature_dim), dtype=np.float16)
        if valid.any():
            features[valid] = self.encode_images(crops, batch_size=batch_size)
        return features, valid

    def encode_texts(self, texts: Sequence[str], batch_size: Optional[int] = None) -> np.ndarray:
        """``[M, 512]`` float16, L2-normalised, for a list of raw strings."""
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
                tokens = self.tokenize(texts[start : start + bs]).to(device)
                with self._autocast(torch):
                    features = self.model.encode_text(tokens)
                chunks.append(np.asarray(features.float().cpu().numpy()))
        normalized = l2_normalize(np.concatenate(chunks, axis=0))
        return normalized.astype(np.float16)

    def token_counts(self, texts: Sequence[str]) -> np.ndarray:
        """BPE token counts *including* the start/end tokens (limit: 77).

        ``SimpleTokenizer.encode`` does not truncate, so values above
        :attr:`context_length` mean the stored embedding saw a truncated text
        (``__call__`` replaces the final token with the end-of-text token).
        """
        encode = getattr(self.tokenize, "encode", None)
        if encode is None:  # pragma: no cover - non-simple tokenizers
            raise RuntimeError(f"tokenizer {type(self.tokenize).__name__} has no encode()")
        return np.asarray([len(encode(str(t))) + 2 for t in texts], dtype=np.int64)

    # -- provenance --------------------------------------------------------
    def metadata(self) -> dict:
        """The frozen ``metadata.json["backbone"]`` record (§5 brief)."""
        return {
            "library": "open_clip_torch",
            "library_version": self.config.library_version,
            "model_name": self.config.model_name,
            "pretrained": self.config.pretrained,
            "checkpoint_path": self.config.checkpoint_path,
            "checkpoint_sha256": self.config.checkpoint_sha256,
            "embedding_dim": self.feature_dim,
            "precision": self.config.precision,
            "resolution": int(self.config.resolution),
            "preprocessing": dict(self.config.preprocessing),
            "tokenizer": {
                "name": self.config.tokenizer_name,
                "context_length": int(self.config.tokenizer_context_length),
            },
        }


def _as_rgb(image):
    Image = _pil()
    if not isinstance(image, Image.Image):
        raise TypeError(f"expected a PIL.Image, got {type(image).__name__}")
    return image if image.mode == "RGB" else image.convert("RGB")


def _probe_feature_dim(model, preprocess, device, torch, fallback: int) -> int:
    """Run one tiny dummy image through the *live* model to read its width.

    open_clip exposes the joint dimensionality inconsistently across versions
    (``model.embed_dim`` is absent on some), so the only backbone-agnostic
    source of truth is the output shape of ``encode_image``.  Any failure falls
    back to the documented V1 width so a probe never blocks encoder creation.
    """
    try:
        Image = _pil()
        tensor = preprocess(Image.new("RGB", (32, 32))).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model.encode_image(tensor)
        dim = int(out.shape[-1])
        if dim > 0:
            return dim
    except Exception:  # pragma: no cover - defensive; a probe must not be fatal
        pass
    return int(fallback)


def build_encoder(
    device: str = "cuda",
    precision: str = "fp16",
    cache_dir: Optional[str | Path] = None,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: str = DEFAULT_PRETRAINED,
    checkpoint_path: Optional[str | Path] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hf_endpoint: Optional[str] = None,
    download_attempts: int = 10,
    download_retry_delay: float = 3.0,
    compute_sha256: bool = True,
) -> ClipEncoder:
    """Create the frozen OpenCLIP ViT-B/32 encoder (``laion2b_s34b_b79k``).

    Steps: resolve the HF endpoint (mirror fallback, see module docstring) ->
    locate/download the checkpoint (resumable, bounded retries) -> build the
    model via ``open_clip.create_model_and_transforms`` -> freeze (eval, no
    grads) -> record the metadata (preprocessing read from the live val
    transform, sha256 of the python checkpoint).

    ``checkpoint_path`` overrides where the *audited* weights file is read
    from (sha256); the model itself is always created from the frozen
    ``pretrained`` tag through the same file when the cache resolves to it.
    """
    if precision not in ("fp16", "fp32"):
        raise ValueError(f"precision must be 'fp16' or 'fp32', got {precision!r}")
    endpoint, endpoint_source = resolve_hf_endpoint(hf_endpoint)

    try:
        import open_clip  # noqa: F401  (import after the endpoint patch)
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "open_clip_torch / torch are required to build the encoder "
            "(declared in pyproject.toml; pip install -e .)"
        ) from exc

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "build_encoder(device='cuda') requested but CUDA is not available; "
            "no silent CPU fallback (feature extraction targets the frozen GPU path)"
        )

    resolved_checkpoint = _ensure_checkpoint(
        model_name,
        pretrained,
        cache_dir,
        checkpoint_path,
        attempts=download_attempts,
        retry_delay=download_retry_delay,
    )
    checkpoint_sha256 = file_sha256(resolved_checkpoint) if compute_sha256 else None

    model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        # load the audited weights directly from the resolved local file so the
        # offline target machine never re-resolves the ``pretrained`` tag over
        # the network; open_clip accepts a checkpoint path here (verified).
        pretrained=str(resolved_checkpoint),
    )
    model = model.to(torch_device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenize = open_clip.get_tokenizer(model_name)
    feature_dim = _probe_feature_dim(
        model, preprocess_val, torch_device, torch, fallback=FEATURE_DIM
    )

    config = ClipEncoderConfig(
        model_name=model_name,
        pretrained=pretrained,
        device=str(torch_device),
        precision=precision,
        batch_size=int(batch_size),
        resolution=DEFAULT_RESOLUTION,
        library_version=str(getattr(open_clip, "__version__", "unknown")),
        checkpoint_path=str(resolved_checkpoint),
        checkpoint_sha256=checkpoint_sha256,
        hf_endpoint=endpoint,
        hf_endpoint_source=endpoint_source,
        tokenizer_name=type(tokenize).__name__,
        tokenizer_context_length=int(getattr(tokenize, "context_length", 77)),
        feature_dim=int(feature_dim),
        preprocessing=describe_preprocessing(preprocess_val),
    )
    return ClipEncoder(model=model, preprocess=preprocess_val, tokenize=tokenize, config=config)
