"""Query-independent Grounding DINO proposal extraction (V2-P2 Proposal-C family).

V2-P2 Phase B asks whether the C1 / C4 reliability phenomena survive a *stronger,
mainstream open-vocabulary pipeline*.  Grounding DINO is text-conditioned by
design, so the single-variable requirement from V2-P1 (protocol P1-A0) pins the
usage: the detector is queried with a **fixed COCO-thing class prompt**, never
with the referring expression.  The resulting bank is therefore conditioned on
the image plus a constant class set - exactly the same conditioning form as the
frozen DETR-R50 (a COCO-80 classifier head) - and swapping DETR -> GDINO changes
the proposal mechanism alone.

Frozen score definition (mirrors :mod:`ccg.data.detr` section 5)
----------------------------------------------------------------
``transformers`` GroundingDino emits ``[Q, L]`` logits (``Q = 900`` queries,
``L = max_text_len`` columns; only the first ``len(prompt tokens)`` columns are
meaningful).  The prompt is tokenized as ``[CLS] c1 . c2 . ... cN . [SEP]``
where ``.`` is the merge separator.  Following the reference post-processing,
the leading ``[CLS]`` column is the global no-text objectness and ``.`` /
``[SEP]`` are not class targets; for each class we keep **its own wordpiece
columns** and define

    score(q) = max over class-token columns of sigmoid(logits[q])

queries are ranked by that score (descending, stable) and truncated to the
frozen bank size ``N = 64``.  No threshold, never padded - identical to DETR.
The winning column maps back to a COCO class label (recorded for diagnostics
only: the same-category regime follows the frozen V1 rule and derives
categories from the highest-IoU COCO *GT* object, never from detector labels).

Coordinate space and sanitisation reuse :mod:`ccg.data.detr` unchanged (with a
row-exact assertion around the label-aligned variant), so Proposal-B and
Proposal-C share one frozen code path.

Determinism: ``eval()`` + ``torch.no_grad()``, one image per call, float32.
The detector never receives the referring expression or any ground truth.
"""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .detr import (  # shared frozen sanitisation / top-N / coordinate code path
    cxcywh_to_xyxy_pixels,
    sanitize_boxes,
    select_top_n,
)

__all__ = [
    "MODEL_ID",
    "MODEL_NAME",
    "PROPOSAL_TYPE",
    "DEFAULT_TOP_N",
    "MERGE_TOKEN",
    "GdinoProposals",
    "GdinoBundle",
    "coco_thing_classes",
    "build_class_prompt",
    "class_token_columns",
    "sanitize_boxes_with_labels",
    "build_gdino_bundle",
    "extract_gdino_proposals",
]

#: Pinned checkpoint identity (revision + sha256 recorded before any use, see
#: results/v2_proposal_robustness/p2_gdino_probe/checkpoint_identity.json).
MODEL_ID = "IDEA-Research/grounding-dino-base"
MODEL_NAME = "grounding_dino_base"
PROPOSAL_TYPE = (
    "query-independent Grounding DINO class-prompt detection "
    "(top-N by max class-token sigmoid prob)"
)

DEFAULT_TOP_N = 64
#: The GDINO merge separator token between class phrases in the prompt.
MERGE_TOKEN = "."
#: The pinned snapshot revision of the local weights directory.
PINNED_REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"


@dataclass
class GdinoProposals:
    """Output contract mirroring :class:`ccg.data.detr.DetrProposals`.

    ``boxes`` is ``[K,4]`` float32 xyxy in original-image pixels ordered by
    descending ``objectness``; ``class_ids`` is ``[K]`` int64 with the winning
    prompt-class index per kept box (diagnostics only, never used for regimes).
    """

    boxes: np.ndarray
    objectness: np.ndarray
    class_ids: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GdinoBundle:
    """Loaded model + processor + frozen prompt layout (class -> token cols)."""

    model: Any
    processor: Any
    device: str
    class_names: List[str]
    class_token_cols: List[List[int]]  # per class: column indices in [0, L)
    prompt_text: str
    n_text_tokens: int
    checkpoint_id: str
    revision: str


# ---------------------------------------------------------------------------
# prompt construction (pure, testable without a GPU)
# ---------------------------------------------------------------------------
def coco_thing_classes(annotation_file: Path) -> List[Tuple[int, str]]:
    """COCO thing categories ``(id, name)`` sorted by id, read from the frozen
    ``instances_train2014.json`` (the same GT universe the V1 audit uses)."""
    data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    cats = [(int(c["id"]), str(c["name"])) for c in data.get("categories", [])]
    return sorted(cats)


def build_class_prompt(class_names: Sequence[str]) -> str:
    """``"c1 . c2 . ... cN ."`` - the GDINO class-prompt convention."""
    return " . ".join(class_names) + " ."


def class_token_columns(
    tokenizer: Any,
    prompt_text: str,
    class_names: Sequence[str],
) -> List[List[int]]:
    """Map every class to its wordpiece columns inside the tokenized prompt.

    Encodes ``prompt_text`` exactly like the processor does (BERT adds
    ``[CLS]``/``[SEP]``), then walks the sequence: between two merge ``.``
    separators lie the tokens of one class phrase.  Verifies, per class, that
    the consumed surface form equals the class name (wordpiece ``##`` glue
    removed); raises on any drift - a changed tokenizer vocabulary must fail
    loudly here, never silently mis-label columns.
    """
    ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    merge_id = tokenizer.convert_tokens_to_ids(MERGE_TOKEN)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    spans: List[List[int]] = []
    current: List[int] = []
    for idx, tok_id in enumerate(ids):
        if idx == 0 and tok_id == cls_id:
            continue  # global no-text objectness column, never a class target
        if tok_id == sep_id:
            break     # trailing [SEP] ends the meaningful columns
        if tok_id == merge_id:
            if current:  # a class phrase ends at every separator
                spans.append(current)
                current = []
            continue
        current.append(idx)
    if current:
        spans.append(current)  # defensive: trailing phrase without separator

    if len(spans) != len(class_names):
        raise ValueError(
            f"prompt produced {len(spans)} class phrases for {len(class_names)} classes; "
            "the merge-token split does not match the class list"
        )
    cols: List[List[int]] = []
    for name, span in zip(class_names, spans):
        surface = "".join(
            tokenizer.convert_ids_to_tokens(ids[i]).replace("##", "") for i in span
        )
        if surface.replace(" ", "") != name.replace(" ", ""):
            raise ValueError(
                f"class {name!r}: prompt tokens {surface!r} disagree with the "
                "standalone class surface form"
            )
        cols.append([int(i) for i in span])
    return cols


# ---------------------------------------------------------------------------
# label-aligned sanitisation (mirrors ccg.data.detr.sanitize_boxes exactly)
# ---------------------------------------------------------------------------
def sanitize_boxes_with_labels(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """``sanitize_boxes`` with a row-aligned ``labels`` carry-along.

    Applies the identical filter order (non-finite, clip, side <= MIN_VALID_SIDE
    via the shared constant, exact-duplicate-first-wins).  Callers must
    cross-check against :func:`ccg.data.detr.sanitize_boxes` with
    :func:`assert_sanitisation_matches` so the two families cannot drift.
    """
    from .detr import MIN_VALID_SIDE

    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not (boxes.shape[0] == scores.shape[0] == labels.shape[0]):
        raise ValueError("boxes/scores/labels length mismatch")
    n0 = int(boxes.shape[0])

    finite = np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
    boxes, scores, labels = boxes[finite], scores[finite], labels[finite]
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
    clipped, scores, labels = clipped[big], scores[big], labels[big]
    n_small = n0 - n_nonfinite - int(clipped.shape[0])

    keys = [tuple(np.round(b, 6).tolist()) for b in clipped]
    seen: set = set()
    keep: List[int] = []
    for idx, key in enumerate(keys):
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    n_dup = int(clipped.shape[0]) - len(keep)
    keep_idx = np.asarray(keep, dtype=np.int64)
    counts = {
        "invalid_finite": int(n_nonfinite),
        "invalid_small": int(n_small),
        "exact_duplicates": int(n_dup),
        "n_input": int(n0),
        "n_kept": int(len(keep)),
    }
    return (
        np.ascontiguousarray(clipped[keep_idx], dtype=np.float32),
        np.ascontiguousarray(scores[keep_idx], dtype=np.float32),
        np.ascontiguousarray(labels[keep_idx], dtype=np.int64),
        counts,
    )


def assert_sanitisation_matches(
    xyxy: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    width: int,
    height: int,
) -> None:
    """Fail loudly if the aligned path drifts from the frozen ``sanitize_boxes``."""
    b_ref, s_ref, c_ref = sanitize_boxes(xyxy, scores, width, height)
    b_got, s_got, _l_got, c_got = sanitize_boxes_with_labels(
        xyxy, scores, labels, width, height
    )
    if not (np.array_equal(b_ref, b_got) and np.array_equal(s_ref, s_got)):
        raise AssertionError("label-aligned sanitisation drifted from ccg.data.detr.sanitize_boxes")
    if c_ref != c_got:
        raise AssertionError(f"sanitize counts differ: {c_ref} vs {c_got}")


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def build_gdino_bundle(
    model_path: str,
    annotation_file: Path,
    device: str = "cuda",
    revision: Optional[str] = None,
) -> GdinoBundle:
    """Load the frozen GDINO checkpoint + processor and pin the class prompt.

    ``model_path`` is the local snapshot directory (downloaded weights; the
    sha256/revision identity was recorded before use).  No device fallback: a
    requested CUDA device that is unavailable raises.
    """
    import torch
    from transformers import AutoProcessor, GroundingDinoForObjectDetection

    torch_dev = torch.device(device)
    if torch_dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    model = GroundingDinoForObjectDetection.from_pretrained(str(model_path))
    processor = AutoProcessor.from_pretrained(str(model_path))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    model.to(torch_dev)

    names = [name for _, name in coco_thing_classes(Path(annotation_file))]
    prompt = build_class_prompt(names)
    cols = class_token_columns(processor.tokenizer, prompt, names)

    return GdinoBundle(
        model=model,
        processor=processor,
        device=str(device),
        class_names=names,
        class_token_cols=cols,
        prompt_text=prompt,
        n_text_tokens=len(processor.tokenizer(prompt, add_special_tokens=True)["input_ids"]),
        checkpoint_id=str(model_path),
        revision=str(revision) if revision else PINNED_REVISION,
    )


# ---------------------------------------------------------------------------
# single-image extraction (driver runs once per image, batch = 1)
# ---------------------------------------------------------------------------
def extract_gdino_proposals(
    image: Any,
    bundle: GdinoBundle,
    top_n: int = DEFAULT_TOP_N,
    autocast: bool = False,
) -> GdinoProposals:
    """Extract the frozen GDINO Proposal-C bank for one RGB image.

    Returns ``GdinoProposals`` with descending-score ``[K,4]`` xyxy boxes
    (never padded), ``objectness`` in ``[0,1]`` (max class-token sigmoid prob),
    and the winning class index per box.  Raises ``RuntimeError`` on a
    zero-propose image.
    """
    import torch
    from .detr import _as_pil

    pil = _as_pil(image)
    width, height = pil.size
    device = torch.device(bundle.device)

    enc = bundle.processor(images=pil, text=bundle.prompt_text, return_tensors="pt")
    kwargs = {k: v.to(device) for k, v in enc.items() if hasattr(v, "to")}

    started = time.perf_counter()
    use_autocast = bool(autocast) and device.type == "cuda"
    with torch.no_grad():
        ctx = torch.autocast("cuda", enabled=True) if use_autocast else nullcontext()
        with ctx:
            outputs = bundle.model(**kwargs)
        logits = outputs.logits[0].float().cpu().numpy()         # [Q, L]
        pred_boxes = outputs.pred_boxes[0].float().cpu().numpy()  # [Q, 4] cxcywh norm
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1e3

    # score per class = max sigmoid over that class's own token columns;
    # objectness = max over classes (isomorphic to DETR's foreground max prob).
    if max(c for cols in bundle.class_token_cols for c in cols) >= logits.shape[1]:
        raise RuntimeError(
            f"logits width {logits.shape[1]} shorter than prompt length"
        )
    pos = 1.0 / (1.0 + np.exp(-logits))  # [Q, L]
    per_class = np.stack(
        [pos[:, cols].max(axis=1) for cols in bundle.class_token_cols], axis=1
    )  # [Q, n_classes]
    scores = per_class.max(axis=1)
    winner = per_class.argmax(axis=1)

    xyxy = cxcywh_to_xyxy_pixels(pred_boxes, width, height)
    boxes, scores2, labels2, counts = sanitize_boxes_with_labels(
        xyxy, scores, winner, width, height
    )
    # hard cross-check against the frozen shared path (never trust the copy)
    assert_sanitisation_matches(xyxy, scores, winner, width, height)
    if boxes.shape[0] == 0:
        raise RuntimeError(
            f"GDINO produced no valid proposals (image {width}x{height}); "
            f"sanitize counts={counts}"
        )
    boxes3, scores3, truncated = select_top_n(boxes, scores2, top_n)
    # same stable descending sort as select_top_n, applied to the aligned labels
    order = np.argsort(-np.asarray(scores2, dtype=np.float32), kind="stable")
    if top_n is not None:
        order = order[: int(top_n)]
    if not np.array_equal(boxes3, boxes[order]):
        raise AssertionError("class-id order drifted from select_top_n ordering")
    class_ids = labels2[order]

    meta: Dict[str, Any] = {
        "model_name": MODEL_NAME,
        "checkpoint_id": bundle.checkpoint_id,
        "revision": bundle.revision,
        "proposal_type": PROPOSAL_TYPE,
        "query_independent": True,
        "prompt_conditioning": "fixed COCO-thing class prompt (never the expression)",
        "original_hw": (int(height), int(width)),
        "num_queries": int(logits.shape[0]),
        "n_classes": int(len(bundle.class_names)),
        "n_text_tokens": int(bundle.n_text_tokens),
        "logits_width": int(logits.shape[1]),
        "sanitize": counts,
        "n_kept_post_sanitize": int(boxes.shape[0]),
        "num_returned": int(boxes3.shape[0]),
        "truncated_to": int(top_n),
        "truncated": bool(truncated),
        "device": str(device),
        "autocast_effective": bool(use_autocast),
        "torch_version": torch.__version__,
        "elapsed_ms": float(elapsed_ms),
    }
    return GdinoProposals(
        boxes=np.ascontiguousarray(boxes3, dtype=np.float32),
        objectness=np.ascontiguousarray(scores3, dtype=np.float32),
        class_ids=np.asarray(class_ids, dtype=np.int64),
        meta=meta,
    )
