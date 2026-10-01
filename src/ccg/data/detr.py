"""Query-independent DETR-R50 proposal extraction (V2-P1 Proposal-B family).

V2-P1 asks a single question: are the two core reliability phenomena (C1
candidate-count degradation, C4 hard-competition semantic amplification) robust
to a *substantially different, query-independent proposal architecture*?  The
frozen reference bank (Proposal-A) is the class-agnostic COCO RPN
(``cache/proposals.h5``, N = 64); this module produces Proposal-B - a frozen,
COCO-pretrained **DETR ResNet-50** set-prediction detector - under exactly the
same ``bank-v1`` contract (:mod:`ccg.data.bank`), so the two families are
directly comparable.

Why DETR-R50 and not Grounding DINO
------------------------------------
Grounding DINO is a *text-conditioned* detector: feeding it the referring
expression would move two variables at once (proposal architecture *and* query
dependence).  DETR-R50 is query-independent by construction - its 100 learned
object queries are conditioned only on the image, never on text or a target
category - so swapping RPN -> DETR changes the *proposal mechanism* alone.

Frozen score definition (protocol section 5)
--------------------------------------------
``facebook/detr-resnet-50`` emits ``[Q, 92]`` class logits per image
(``Q = 100`` queries).  The classification head is ``num_labels + 1 = 92`` wide:

* ``id2label`` index ``0`` = ``"N/A"`` (a COCO category-id alignment placeholder,
  never a training target, so it carries negligible probability);
* indices ``1 .. 90`` = the 90 COCO categories;
* the final column, index ``91`` = ``num_labels`` = the dedicated **no-object**
  class DETR trains every empty query against.

For each query we take the **maximum foreground class probability**, i.e. the
softmax over the 91 COCO-side columns with the trailing no-object column
excluded::

    score(q) = max_j softmax(logits[q])[j]   for j in 0 .. num_labels-1

Queries are then ranked by that score (descending, stable) and truncated to the
frozen bank size ``N = 64``.  No score threshold is applied (the RPN keeps a
fixed top-N too), and an image with fewer than 64 surviving sanitized boxes is
reported with its real count - never padded.

Coordinate space
----------------
``pred_boxes`` are ``[cx, cy, w, h]`` normalised to ``[0, 1]`` with respect to
the resized image.  The processor resizes shortest-edge 800 / longest-cap 1333
**aspect-preservingly** and, at batch size 1, adds no padding, so a normalised
coordinate equals the coordinate in original-image space scaled by ``(W, H)``.
This is exactly the inverse of ``DetrImageProcessor.post_process_object_detection``
and is why extraction runs one image at a time (a padded batch would break the
uniform-scale assumption).  Boxes are converted to absolute ``xyxy`` and clipped
to the image.

Box sanitisation (protocol section 6)
-------------------------------------
After conversion: clip to bounds, drop ``width <= 1`` / ``height <= 1`` /
non-finite, and drop *exact* duplicate ``xyxy`` rows (first, highest-score wins).
Near-duplicate / pairwise redundancy is only *measured and recorded* (no
result-driven NMS in the first version).

Determinism: ``eval()`` + ``torch.no_grad()``, one image per call, float32
outputs, a stable descending-score sort.  The detector never receives the
referring expression, a target category, or any ground truth.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = [
    "MODEL_ID",
    "MODEL_NAME",
    "PROPOSAL_TYPE",
    "DEFAULT_TOP_N",
    "DEFAULT_MIN_SIZE",
    "DEFAULT_MAX_SIZE",
    "MIN_VALID_SIDE",
    "DetrProposals",
    "DetrBundle",
    "build_detr_bundle",
    "extract_detr_proposals",
    "foreground_scores",
    "cxcywh_to_xyxy_pixels",
    "sanitize_boxes",
    "select_top_n",
    "near_duplicate_stats",
    "extract_detr_proposals_batch",
]

#: The frozen COCO-pretrained checkpoint identity (never swapped after results).
MODEL_ID = "facebook/detr-resnet-50"
#: Detector-family label recorded in every bank ``meta`` / root attr.
MODEL_NAME = "detr_resnet50"
PROPOSAL_TYPE = "query-independent DETR set-prediction object queries (top-N by foreground prob)"

DEFAULT_TOP_N = 64
DEFAULT_MIN_SIZE = 800
DEFAULT_MAX_SIZE = 1333
#: A proposal thinner than this (post-clip) is dropped by sanitisation.
MIN_VALID_SIDE = 1.0


@dataclass
class DetrProposals:
    """Output contract of :func:`extract_detr_proposals` (mirrors RPNProposals).

    ``boxes`` is ``[K, 4]`` float32 ``xyxy`` in original-image pixels, ordered by
    descending ``objectness``; ``objectness`` is ``[K]`` float32 in ``[0, 1]``;
    ``meta`` carries the frozen bookkeeping keys (see :func:`extract_detr_proposals`).
    """

    boxes: np.ndarray
    objectness: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetrBundle:
    """Loaded model + processor + resolved classification constants (frozen)."""

    model: Any
    processor: Any
    device: str
    num_labels: int          # 91 (COCO-side classes incl. the "N/A" placeholder)
    num_queries: int         # 100
    no_object_index: int     # == num_labels (the trailing background column)
    checkpoint_id: str
    revision: str

    @property
    def foreground_columns(self) -> slice:
        """Columns that count as foreground: every class except no-object."""
        return slice(0, self.num_labels)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def build_detr_bundle(
    model_id: str = MODEL_ID,
    device: str = "cuda",
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
) -> DetrBundle:
    """Load the frozen DETR-R50 detector + image processor (eval, no grad).

    No device fallback: a requested CUDA device that is unavailable raises.  The
    model is put in ``eval()`` with every parameter ``requires_grad_(False)`` -
    nothing here is trained in Route F.
    """
    import torch
    from transformers import DetrForObjectDetection, DetrImageProcessor

    kwargs: Dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if revision is not None:
        kwargs["revision"] = str(revision)

    torch_dev = torch.device(device)
    if torch_dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    model = DetrForObjectDetection.from_pretrained(model_id, **kwargs)
    processor = DetrImageProcessor.from_pretrained(model_id, **kwargs)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(torch_dev)

    num_labels = int(model.config.num_labels)
    num_queries = int(getattr(model.config, "num_queries", 100))
    # The classification head is num_labels + 1 wide; the last column is no-object.
    no_object_index = num_labels

    # Resolve the pinned commit this snapshot came from (best effort; None if
    # the offline cache does not expose it).
    resolved_revision = _snapshot_revision(model, cache_dir)

    return DetrBundle(
        model=model,
        processor=processor,
        device=str(device),
        num_labels=num_labels,
        num_queries=num_queries,
        no_object_index=no_object_index,
        checkpoint_id=str(model_id),
        revision=resolved_revision or (str(revision) if revision else "main"),
    )


def _snapshot_revision(model: Any, cache_dir: Optional[str]) -> Optional[str]:
    """Best-effort pinned commit hash of the loaded snapshot."""
    try:
        from transformers.utils import hf_hub_download  # noqa: F401
    except Exception:
        return None
    name = getattr(model, "name_or_path", None)
    if not name:
        return None
    base = Path(cache_dir) if cache_dir else Path(
        __import__("os").environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    ) / "hub"
    ref = base / f"models--{name.replace('/', '--')}" / "refs" / "main"
    try:
        if ref.exists():
            text = ref.read_text(encoding="utf-8").strip()
            if len(text) == 40:
                return text
    except OSError:
        return None
    return None


# ---------------------------------------------------------------------------
# pure numpy / torch helpers (importable in tests without a GPU)
# ---------------------------------------------------------------------------
def foreground_scores(logits: Any, num_labels: int) -> np.ndarray:
    """Max foreground class probability per query, excluding the no-object column.

    ``logits`` is a ``[Q, C]`` array (torch tensor or numpy), ``C == num_labels + 1``
    with column ``num_labels`` the no-object class.  Returns ``[Q]`` float64.
    """
    arr = _to_numpy_float(logits)
    if arr.ndim != 2:
        raise ValueError(f"expected [Q, C] logits, got shape {arr.shape}")
    c = arr.shape[1]
    if c != num_labels + 1:
        raise ValueError(
            f"logits width {c} disagrees with num_labels+1 ({num_labels + 1}); "
            "the no-object column is the final class"
        )
    exp = np.exp(arr - arr.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs[:, :num_labels].max(axis=1)


def cxcywh_to_xyxy_pixels(
    boxes: Any,
    width: int,
    height: int,
) -> np.ndarray:
    """Normalised ``[cx, cy, w, h]`` -> absolute ``xyxy`` pixels (no rescale padding).

    The ``[Q,4]`` result is scaled per axis by ``[width, height, width, height]``
    (x-columns by width, y-columns by height) - the exact inverse of
    ``DetrImageProcessor.post_process_object_detection``.
    """
    arr = _to_numpy_float(boxes)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"expected [Q,4] cxcywh, got {arr.shape}")
    if width <= 0 or height <= 0:
        raise ValueError(f"degenerate image size {width}x{height}")
    cx, cy, w, h = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    x1 = (cx - 0.5 * w) * float(width)
    y1 = (cy - 0.5 * h) * float(height)
    x2 = (cx + 0.5 * w) * float(width)
    y2 = (cy + 0.5 * h) * float(height)
    return np.stack([x1, y1, x2, y2], axis=1)


def _drop_exact_duplicates(boxes: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """Keep the first (highest-score) row of every exactly-equal ``xyxy`` box.

    Callers must pass rows already ordered so "first" is deterministic (this
    helper preserves input order; the sort happens around it in
    :func:`select_top_n`).  Returns ``(boxes, scores, n_duplicates_removed)``.
    """
    if boxes.shape[0] == 0:
        return boxes, scores, 0
    # quantise to a stable string key so float noise never creates a "new" box
    keys = [tuple(np.round(b, 6).tolist()) for b in boxes]
    seen: set = set()
    keep: list = []
    for idx, key in enumerate(keys):
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    removed = int(boxes.shape[0] - len(keep))
    keep_idx = np.asarray(keep, dtype=np.int64)
    return boxes[keep_idx], scores[keep_idx], removed


def sanitize_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Clip to bounds; drop degenerate / non-finite / exact-duplicate boxes.

    Returns ``(boxes, scores, counts)`` where ``counts`` reports
    ``invalid_finite``, ``invalid_small`` (post-clip ``w<=1`` or ``h<=1``) and
    ``exact_duplicates``.  Order-preserving before the caller re-sorts.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(
            f"boxes/scores length mismatch ({boxes.shape[0]} vs {scores.shape[0]})"
        )
    n0 = int(boxes.shape[0])
    finite = np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
    boxes, scores = boxes[finite], scores[finite]
    n_nonfinite = n0 - int(boxes.shape[0])

    clipped = boxes.copy()
    if clipped.shape[0]:
        np.clip(
            clipped, 0.0,
            np.array([width, height, width, height], dtype=np.float64),
            out=clipped,
        )
    w = clipped[:, 2] - clipped[:, 0]
    h = clipped[:, 3] - clipped[:, 1]
    big = (w > MIN_VALID_SIDE) & (h > MIN_VALID_SIDE)
    clipped, scores = clipped[big], scores[big]
    n_small = n0 - n_nonfinite - int(clipped.shape[0])

    clipped, scores, n_dup = _drop_exact_duplicates(clipped, scores)
    counts = {
        "invalid_finite": int(n_nonfinite),
        "invalid_small": int(n_small),
        "exact_duplicates": int(n_dup),
        "n_input": int(n0),
        "n_kept": int(clipped.shape[0]),
    }
    return (
        np.ascontiguousarray(clipped, dtype=np.float32),
        np.ascontiguousarray(scores, dtype=np.float32),
        counts,
    )


def select_top_n(
    boxes: np.ndarray,
    scores: np.ndarray,
    top_n: Optional[int] = DEFAULT_TOP_N,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Stable descending-score sort + top-N truncation (never padded)."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    truncated = top_n is not None and order.size > int(top_n)
    if top_n is not None:
        order = order[: int(top_n)]
    return boxes[order], scores[order], bool(truncated)


def near_duplicate_stats(boxes: np.ndarray, iou_high: float = 0.9) -> Dict[str, float]:
    """Fraction of proposal pairs above ``iou_high`` (recorded, never filtered)."""
    from .proposals import iou_matrix

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    n = int(boxes.shape[0])
    if n < 2:
        return {"n_pairs": 0, "frac_pairs_gt_high": 0.0}
    ious = iou_matrix(boxes, boxes)
    iu = np.triu_indices(n, k=1)
    pair = ious[iu]
    return {
        "n_pairs": int(pair.size),
        "frac_pairs_gt_high": float(np.mean(pair > iou_high)) if pair.size else 0.0,
    }


def _to_numpy_float(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().to("cpu").double().numpy()
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# single-image extraction (the driver runs this once per image, batch = 1)
# ---------------------------------------------------------------------------
def extract_detr_proposals(
    image: Any,
    bundle: DetrBundle,
    top_n: int = DEFAULT_TOP_N,
    autocast: bool = False,
) -> DetrProposals:
    """Extract the frozen DETR Proposal-B bank for one image.

    ``image`` is a PIL RGB image (or a path).  Returns ``DetrProposals`` with
    ``boxes`` ``[K,4]`` xyxy float32 (descending score), ``objectness`` ``[K]``
    float32, and a ``meta`` dict.  Raises ``RuntimeError`` on a zero-propose image.
    """
    import torch

    pil = _as_pil(image)
    width, height = pil.size
    device = torch.device(bundle.device)

    enc = bundle.processor(images=pil, return_tensors="pt")
    pixel_values = enc["pixel_values"].to(device)
    pixel_mask = enc.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(device)

    started = time.perf_counter()
    use_autocast = bool(autocast) and device.type == "cuda"
    with torch.no_grad():
        ctx = torch.autocast("cuda", enabled=True) if use_autocast else nullcontext()
        with ctx:
            outputs = bundle.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        logits = outputs.logits[0].float().cpu().numpy()      # [Q, 92]
        pred_boxes = outputs.pred_boxes[0].float().cpu().numpy()  # [Q, 4] cxcywh norm
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1e3

    scores = foreground_scores(logits, bundle.num_labels)          # [Q]
    xyxy = cxcywh_to_xyxy_pixels(pred_boxes, width, height)         # [Q,4] abs px
    boxes, scores2, counts = sanitize_boxes(xyxy, scores, width, height)
    if boxes.shape[0] == 0:
        raise RuntimeError(
            f"DETR produced no valid proposals (image {width}x{height}); "
            f"sanitize counts={counts}"
        )
    boxes, scores3, truncated = select_top_n(boxes, scores2, top_n)

    meta: Dict[str, Any] = {
        "model_name": MODEL_NAME,
        "checkpoint_id": bundle.checkpoint_id,
        "revision": bundle.revision,
        "proposal_type": PROPOSAL_TYPE,
        "original_hw": (int(height), int(width)),
        "num_queries": int(bundle.num_queries),
        "num_labels": int(bundle.num_labels),
        "no_object_index": int(bundle.no_object_index),
        "min_size": DEFAULT_MIN_SIZE,
        "max_size": DEFAULT_MAX_SIZE,
        "sanitize": counts,
        "n_kept_post_sanitize": int(boxes.shape[0]),
        "num_returned": int(boxes.shape[0]),
        "truncated_to": int(top_n),
        "truncated": bool(truncated),
        "device": str(device),
        "autocast_effective": bool(use_autocast),
        "torch_version": torch.__version__,
        "elapsed_ms": float(elapsed_ms),
    }
    return DetrProposals(
        boxes=np.ascontiguousarray(boxes, dtype=np.float32),
        objectness=np.ascontiguousarray(scores3, dtype=np.float32),
        meta=meta,
    )


def extract_detr_proposals_batch(
    images: "list[Any]",
    sizes: "list[tuple[int, int]]",
    bundle: DetrBundle,
    top_n: int = DEFAULT_TOP_N,
    autocast: bool = False,
) -> "list[DetrProposals]":
    """Frozen DETR Proposal-B bank for a padded batch (one forward call).

    ``images`` are PIL RGB images; ``sizes`` are their ``(width, height)``.
    The processor resizes each aspect-preservingly and pads to the batch max,
    but DETR's normalised ``pred_boxes`` are still relative to each image's own
    (unpadded) content, so scaling by that image's original ``(W, H)`` is exact
    (this is ``post_process_object_detection``'s documented batch behaviour).
    Returns one :class:`DetrProposals` per input image, in the same order.
    """
    import torch

    if len(images) != len(sizes):
        raise ValueError("images and sizes must be aligned")
    if not images:
        return []
    device = torch.device(bundle.device)
    pils = [_as_pil(im) for im in images]
    enc = bundle.processor(images=pils, return_tensors="pt")
    pixel_values = enc["pixel_values"].to(device)
    pixel_mask = enc.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(device)

    started = time.perf_counter()
    use_autocast = bool(autocast) and device.type == "cuda"
    with torch.no_grad():
        ctx = torch.autocast("cuda", enabled=True) if use_autocast else nullcontext()
        with ctx:
            outputs = bundle.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        logits = outputs.logits.float().cpu().numpy()        # [B, Q, C]
        pred_boxes = outputs.pred_boxes.float().cpu().numpy()  # [B, Q, 4]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1e3

    results: "list[DetrProposals]" = []
    for i, (width, height) in enumerate(sizes):
        scores = foreground_scores(logits[i], bundle.num_labels)
        xyxy = cxcywh_to_xyxy_pixels(pred_boxes[i], int(width), int(height))
        boxes, scores2, counts = sanitize_boxes(xyxy, scores, int(width), int(height))
        if boxes.shape[0] == 0:
            raise RuntimeError(
                f"DETR produced no valid proposals (image {width}x{height}); "
                f"sanitize counts={counts}"
            )
        boxes, scores3, truncated = select_top_n(boxes, scores2, top_n)
        meta: Dict[str, Any] = {
            "model_name": MODEL_NAME,
            "checkpoint_id": bundle.checkpoint_id,
            "revision": bundle.revision,
            "proposal_type": PROPOSAL_TYPE,
            "original_hw": (int(height), int(width)),
            "num_queries": int(bundle.num_queries),
            "num_labels": int(bundle.num_labels),
            "no_object_index": int(bundle.no_object_index),
            "min_size": DEFAULT_MIN_SIZE,
            "max_size": DEFAULT_MAX_SIZE,
            "sanitize": counts,
            "n_kept_post_sanitize": int(boxes.shape[0]),
            "num_returned": int(boxes.shape[0]),
            "truncated_to": int(top_n),
            "truncated": bool(truncated),
            "device": str(device),
            "autocast_effective": bool(use_autocast),
            "torch_version": torch.__version__,
            "batch_size": int(len(images)),
            "elapsed_ms_batch": float(elapsed_ms),
        }
        results.append(
            DetrProposals(
                boxes=np.ascontiguousarray(boxes, dtype=np.float32),
                objectness=np.ascontiguousarray(scores3, dtype=np.float32),
                meta=meta,
            )
        )
    return results


def _as_pil(image: Any) -> Any:
    from PIL import Image

    if hasattr(image, "convert") and hasattr(image, "size"):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(str(image)) as handle:
            return handle.convert("RGB")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")
