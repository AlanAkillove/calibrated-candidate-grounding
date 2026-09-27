"""Proposal banks: geometry, target assignment, storage and quality audits.

Sections 4 and 6 of the protocol.  Three concerns live here, all of them pure
CPU / numpy (h5py is imported lazily so an environment without it can still use
the geometry and audit functions):

* geometry - :func:`iou_matrix`, :func:`box_iou` and the ``xyxy``/``xywh``
  converters (banks are always stored ``xyxy``);
* target assignment - :func:`assign_target` picks the single highest-IoU proposal
  as target and reports the *equivalent* proposals that must be removed so a
  candidate set can never contain two correct answers;
* storage / audits - HDF5 (preferred) or NPZ backends for
  :class:`~ccg.data.types.ProposalBank` plus the proposal-quality statistics the
  audit script reports before any grounding experiment starts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from .types import ProposalBank, ReferringExample

__all__ = [
    "DEFAULT_IOU_THRESH",
    "DEFAULT_AUDIT_TOP_KS",
    "box_iou",
    "box_area",
    "iou_matrix",
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "TargetAssignment",
    "assign_target",
    "remove_proposals",
    "target_max_ious",
    "proposal_recall",
    "natural_miss_rate",
    "iou_distribution",
    "summarize_proposal_quality",
    "audit_examples",
    "hdf5_available",
    "write_bank",
    "write_banks",
    "read_bank",
    "iter_banks",
    "read_all_banks",
    "list_image_ids",
    "read_features",
    "feature_ref",
]

#: Protocol section 4: a proposal is the target iff IoU >= 0.5 (and maximal).
DEFAULT_IOU_THRESH = 0.5
#: Recall is audited at @32 and @64 (section 4).
DEFAULT_AUDIT_TOP_KS: tuple[int, ...] = (32, 64)

_H5_SUFFIXES = (".h5", ".hdf5", ".hdf", ".hd5")
_EPS = 1e-10


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def box_area(boxes: np.ndarray) -> np.ndarray:
    """Areas of ``xyxy`` boxes, ``[N]`` (a single box is accepted too)."""
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"expected boxes with shape [N,4], got {arr.shape}")
    return np.maximum(arr[:, 2] - arr[:, 0], 0.0) * np.maximum(arr[:, 3] - arr[:, 1], 0.0)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """``[x, y, w, h]`` -> ``[x1, y1, x2, y2]`` (RefCOCO json -> internal format)."""
    arr = np.asarray(boxes, dtype=np.float32)
    single = arr.ndim == 1
    arr = arr.reshape(1, 4) if single else arr
    if arr.shape[1] != 4:
        raise ValueError(f"expected [*,4] boxes, got {arr.shape}")
    out = arr.copy()
    out[:, 2] = arr[:, 0] + arr[:, 2]
    out[:, 3] = arr[:, 1] + arr[:, 3]
    return out.reshape(4) if single else out


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """``[x1, y1, x2, y2]`` -> ``[x, y, w, h]`` (annotation writers / COCO json)."""
    arr = np.asarray(boxes, dtype=np.float32)
    single = arr.ndim == 1
    arr = arr.reshape(1, 4) if single else arr
    out = arr.copy()
    out[:, 2] = arr[:, 2] - arr[:, 0]
    out[:, 3] = arr[:, 3] - arr[:, 1]
    return out.reshape(4) if single else out


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU, ``[len(a), len(b)]``, pure numpy."""
    a = np.asarray(boxes_a, dtype=np.float32)
    b = np.asarray(boxes_b, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 4 or b.ndim != 2 or b.shape[1] != 4:
        raise ValueError(f"expected [N,4] and [M,4] xyxy boxes, got {a.shape} and {b.shape}")
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    left = np.maximum(a[:, None, 0], b[None, :, 0])
    right = np.minimum(a[:, None, 2], b[None, :, 2])
    top = np.maximum(a[:, None, 1], b[None, :, 1])
    bottom = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(right - left, 0.0) * np.maximum(bottom - top, 0.0)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return (inter / np.maximum(union, _EPS)).astype(np.float32, copy=False)


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between two ``xyxy`` boxes."""
    return float(iou_matrix(np.asarray(box_a, dtype=np.float32).reshape(1, 4),
                            np.asarray(box_b, dtype=np.float32).reshape(1, 4))[0, 0])


# ---------------------------------------------------------------------------
# target assignment
# ---------------------------------------------------------------------------
@dataclass
class TargetAssignment:
    """Result of :func:`assign_target` for one (image, expression) pair."""

    image_id: int
    target_proposal_idx: Optional[int]
    best_iou: float
    equivalent_indices: np.ndarray  # every proposal with IoU >= threshold
    to_remove: np.ndarray  # equivalent_indices minus the target itself
    iou_thresh: float

    @property
    def is_miss(self) -> bool:
        """``True`` for a natural proposal miss (no proposal is good enough)."""
        return self.target_proposal_idx is None

    @property
    def num_equivalent(self) -> int:
        return int(self.equivalent_indices.size)

    def to_dict(self) -> dict:
        return {
            "image_id": int(self.image_id),
            "target_proposal_idx": (
                None if self.target_proposal_idx is None else int(self.target_proposal_idx)
            ),
            "best_iou": float(self.best_iou),
            "iou_thresh": float(self.iou_thresh),
            "equivalent_indices": self.equivalent_indices.astype(int).tolist(),
            "to_remove": self.to_remove.astype(int).tolist(),
            "is_miss": self.is_miss,
        }


def assign_target(
    bank: ProposalBank,
    gt_box: np.ndarray,
    iou_thresh: float = DEFAULT_IOU_THRESH,
) -> TargetAssignment:
    """Pick the unique target proposal and the equivalents that must be removed.

    Rule (protocol section 4): if ``max_i IoU(p_i, b*) >= iou_thresh`` the
    *argmax* proposal is the single target ``c*``; every other proposal with
    ``IoU >= iou_thresh`` is an equivalent correct answer and has to be removed
    from the pool, otherwise a candidate set would contain two valid targets and
    "accuracy" would depend on an arbitrary tie-break.

    Ties in IoU are broken by the smallest row index (deterministic) and, when
    the bank carries objectness, by the higher objectness score first - the
    detector's own preference is the least surprising tie-break.

    When no proposal reaches the threshold the target is ``None``; that example
    belongs to the ``natural_omission`` evaluation split and must be reported
    separately from synthetic omission (section 6).
    """
    gt = np.asarray(gt_box, dtype=np.float32).reshape(1, 4)
    ious = iou_matrix(gt, bank.boxes)[0]
    if ious.size == 0:
        raise ValueError(f"bank for image {bank.image_id} has no proposals")
    best_iou = float(ious.max())
    equivalent = np.nonzero(ious >= float(iou_thresh))[0].astype(np.int64)
    if equivalent.size == 0:
        return TargetAssignment(
            image_id=bank.image_id,
            target_proposal_idx=None,
            best_iou=best_iou,
            equivalent_indices=equivalent,
            to_remove=np.empty(0, dtype=np.int64),
            iou_thresh=float(iou_thresh),
        )
    # deterministic argmax: IoU desc, then objectness desc, then row index asc
    tied = equivalent[ious[equivalent] >= best_iou - _EPS]
    if bank.objectness is not None and tied.size > 1:
        tied = tied[np.argsort(-bank.objectness[tied], kind="stable")]
    target = int(tied[0])
    to_remove = np.setdiff1d(equivalent, [target]).astype(np.int64)
    return TargetAssignment(
        image_id=bank.image_id,
        target_proposal_idx=target,
        best_iou=best_iou,
        equivalent_indices=equivalent,
        to_remove=to_remove,
        iou_thresh=float(iou_thresh),
    )


def remove_proposals(bank: ProposalBank, indices: Sequence[int]) -> tuple[ProposalBank, np.ndarray]:
    """Drop proposals ``indices`` from ``bank``; also return the row remap.

    ``remap[old_row] = new_row`` (``-1`` for dropped rows) so candidate indices
    built against the original bank can be translated into the cleaned bank.
    """
    drop = np.zeros(bank.N, dtype=bool)
    for index in np.asarray(indices, dtype=np.int64).reshape(-1):
        if not 0 <= int(index) < bank.N:
            raise ValueError(f"index {index} out of range for bank of size {bank.N}")
        drop[int(index)] = True
    keep = np.nonzero(~drop)[0]
    if keep.size == 0:
        raise ValueError("removal would empty the proposal bank")
    remap = np.full(bank.N, -1, dtype=np.int64)
    remap[keep] = np.arange(keep.size, dtype=np.int64)
    cleaned = ProposalBank(
        image_id=bank.image_id,
        boxes=bank.boxes[keep],
        objectness=bank.objectness[keep],
        gt_ious=None if bank.gt_ious is None else bank.gt_ious[keep],
        gt_assignment=None if bank.gt_assignment is None else bank.gt_assignment[keep],
        feature_ref=None if bank.feature_ref is None else f"{bank.feature_ref}[cleaned]",
    )
    return cleaned, remap


# ---------------------------------------------------------------------------
# quality audits (pure functions; scripts/audit_proposals.py calls these)
# ---------------------------------------------------------------------------
def _bank_boxes(bank: ProposalBank | np.ndarray, top_k: Optional[int] = None) -> np.ndarray:
    boxes = bank.boxes if isinstance(bank, ProposalBank) else np.asarray(bank, dtype=np.float32)
    boxes = boxes.reshape(-1, 4)
    if top_k is not None and boxes.shape[0] > top_k:
        if isinstance(bank, ProposalBank):
            order = bank.top_k_order(top_k)
            return boxes[order]
        return boxes[:top_k]
    return boxes


def target_max_ious(
    gt_boxes: Sequence[np.ndarray],
    bank_boxes: Sequence[np.ndarray],
    top_k: Optional[int] = None,
) -> np.ndarray:
    """Per-example ``max_i IoU(p_i, b*)`` over (optionally truncated) banks."""
    if len(gt_boxes) != len(bank_boxes):
        raise ValueError(
            f"gt_boxes ({len(gt_boxes)}) and bank_boxes ({len(bank_boxes)}) must be aligned"
        )
    out = np.empty(len(gt_boxes), dtype=np.float32)
    for i, (gt, bank) in enumerate(zip(gt_boxes, bank_boxes)):
        boxes = _bank_boxes(bank, top_k)
        out[i] = float(iou_matrix(np.asarray(gt, dtype=np.float32).reshape(1, 4), boxes).max())
    return out


def proposal_recall(
    gt_boxes: Sequence[np.ndarray],
    bank_boxes: Sequence[np.ndarray],
    iou_thresh: float = DEFAULT_IOU_THRESH,
    top_k: Optional[int] = None,
) -> float:
    """Fraction of expressions whose target is covered by the (top-k) bank."""
    ious = target_max_ious(gt_boxes, bank_boxes, top_k=top_k)
    if ious.size == 0:
        raise ValueError("proposal_recall() called with no examples")
    return float(np.mean(ious >= iou_thresh))


def natural_miss_rate(
    gt_boxes: Sequence[np.ndarray],
    bank_boxes: Sequence[np.ndarray],
    iou_thresh: float = DEFAULT_IOU_THRESH,
    top_k: Optional[int] = None,
) -> float:
    """Fraction of expressions with ``max_i IoU(p_i, b*) < iou_thresh`` (section 6)."""
    return 1.0 - proposal_recall(gt_boxes, bank_boxes, iou_thresh=iou_thresh, top_k=top_k)


def iou_distribution(
    ious: Sequence[float],
    bin_edges: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
) -> Dict[str, object]:
    """Histogram + quantiles of per-example max-IoU values."""
    values = np.asarray(list(ious), dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("iou_distribution() called with no values")
    edges = np.asarray(list(bin_edges), dtype=np.float64)
    counts, edges = np.histogram(values, bins=edges)
    return {
        "num_examples": int(values.size),
        "bin_edges": edges.tolist(),
        "counts": counts.astype(int).tolist(),
        "fractions": (counts / float(values.size)).tolist(),
        "mean": float(values.mean()),
        "quantiles": {
            f"q{int(q * 100)}": float(np.quantile(values, q))
            for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "frac_ge_0.5": float(np.mean(values >= 0.5)),
    }


def summarize_proposal_quality(
    gt_boxes: Sequence[np.ndarray],
    bank_boxes: Sequence[np.ndarray],
    iou_thresh: float = DEFAULT_IOU_THRESH,
    top_ks: Sequence[int] = DEFAULT_AUDIT_TOP_KS,
    bin_edges: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
) -> Dict[str, object]:
    """The proposal audit of section 4: recall@K, natural miss rate, IoU spread."""
    if len(gt_boxes) == 0:
        raise ValueError("summarize_proposal_quality() called with no examples")
    summary: Dict[str, object] = {
        "num_examples": int(len(gt_boxes)),
        "iou_thresh": float(iou_thresh),
        "proposal_counts": [int(np.asarray(b).reshape(-1, 4).shape[0]) for b in bank_boxes],
        "recall_by_k": {},
        "natural_miss_rate_by_k": {},
    }
    for k in top_ks:
        ious = target_max_ious(gt_boxes, bank_boxes, top_k=int(k))
        summary["recall_by_k"][str(int(k))] = float(np.mean(ious >= iou_thresh))
        summary["natural_miss_rate_by_k"][str(int(k))] = float(np.mean(ious < iou_thresh))
    full_ious = target_max_ious(gt_boxes, bank_boxes)
    summary["max_iou_distribution"] = iou_distribution(full_ious, bin_edges=bin_edges)
    summary["num_equivalent_target_proposals"] = _count_equivalents(
        gt_boxes, bank_boxes, iou_thresh
    )
    return summary


def _count_equivalents(
    gt_boxes: Sequence[np.ndarray], bank_boxes: Sequence[np.ndarray], iou_thresh: float
) -> Dict[str, float]:
    """How many images carry >1 proposal above threshold (ambiguous targets)."""
    counts = []
    for gt, bank in zip(gt_boxes, bank_boxes):
        boxes = _bank_boxes(bank)
        ious = iou_matrix(np.asarray(gt, dtype=np.float32).reshape(1, 4), boxes)[0]
        counts.append(int(np.sum(ious >= iou_thresh)))
    counts_arr = np.asarray(counts, dtype=np.int64)
    return {
        "mean": float(counts_arr.mean()),
        "frac_examples_with_multiple": float(np.mean(counts_arr > 1)),
        "max": int(counts_arr.max()),
    }


def audit_examples(
    examples: Sequence[ReferringExample],
    banks: Mapping[int, ProposalBank] | Sequence[ProposalBank],
    iou_thresh: float = DEFAULT_IOU_THRESH,
    top_ks: Sequence[int] = DEFAULT_AUDIT_TOP_KS,
) -> Dict[str, object]:
    """Same audit as :func:`summarize_proposal_quality`, driven by real records.

    ``banks`` is either ``{image_id: ProposalBank}`` or a sequence aligned with
    ``examples``.  Examples whose image has no bank are counted in
    ``missing_banks`` and *not* silently dropped from the denominator.
    """
    if not isinstance(banks, Mapping):
        banks_by_image = {int(b.image_id): b for b in banks}
    else:
        banks_by_image = {int(k): v for k, v in banks.items()}
    gt_boxes: List[np.ndarray] = []
    bank_boxes: List[np.ndarray] = []
    missing = 0
    for example in examples:
        bank = banks_by_image.get(int(example.image_id))
        if bank is None:
            missing += 1
            continue
        gt_boxes.append(example.gt_box)
        bank_boxes.append(bank.boxes)
    report = summarize_proposal_quality(
        gt_boxes, bank_boxes, iou_thresh=iou_thresh, top_ks=top_ks
    )
    report["missing_banks"] = missing
    report["num_requested_examples"] = int(len(examples))
    return report


# ---------------------------------------------------------------------------
# storage backends
# ---------------------------------------------------------------------------
def hdf5_available() -> bool:
    """``True`` when h5py can be imported (the preferred random-access cache)."""
    try:
        import h5py  # noqa: F401
    except Exception:  # pragma: no cover - environment dependent
        return False
    return True


def feature_ref(path: str | Path, image_id: int) -> str:
    """Canonical handle stored in ``ProposalBank.feature_ref`` (never in memory)."""
    return f"cache://{Path(path).as_posix()}#/images/{int(image_id)}/features"


def _detect_backend(path: str | Path, backend: Optional[str]) -> str:
    if backend is not None:
        if backend not in ("hdf5", "npz"):
            raise ValueError(f"unknown backend {backend!r}; expected 'hdf5' or 'npz'")
        return backend
    suffix = Path(path).suffix.lower()
    if suffix in _H5_SUFFIXES:
        return "hdf5"
    if suffix == ".npz":
        return "npz"
    raise ValueError(f"cannot infer a cache backend from {path!r} (use backend='hdf5' or 'npz')")


def _bank_payload(bank: ProposalBank) -> Dict[str, np.ndarray]:
    payload = {
        "boxes": np.asarray(bank.boxes, dtype=np.float32),
        "objectness": np.asarray(bank.objectness, dtype=np.float32),
    }
    if bank.gt_ious is not None:
        payload["gt_iou"] = np.asarray(bank.gt_ious, dtype=np.float32)
    if bank.gt_assignment is not None:
        payload["gt_assignment"] = np.asarray(bank.gt_assignment, dtype=np.int64)
    return payload


def write_bank(
    path: str | Path,
    bank: ProposalBank,
    features: Optional[np.ndarray] = None,
    *,
    mode: str = "a",
    backend: Optional[str] = None,
) -> Path:
    """Write one :class:`ProposalBank` into the cache (one group per image).

    ``features`` (``[N, D]`` crop embeddings) is optional and, when present, is
    stored as ``float16`` to honour the <3 GB cache budget; it is *never*
    mirrored into the dataclass, only referenced through ``feature_ref``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chosen = _detect_backend(path, backend)
    if features is not None:
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[0] != bank.N:
            raise ValueError(
                f"features must have shape [N,D] with N={bank.N}, got {features.shape}"
            )
    if chosen == "hdf5":
        _h5_write(path, bank, features, mode=mode)
    else:
        _npz_write(path, bank, features, mode=mode)
    return path


def write_banks(
    path: str | Path,
    banks: Iterable[ProposalBank],
    features: Optional[Mapping[int, np.ndarray]] = None,
    *,
    mode: str = "w",
    backend: Optional[str] = None,
) -> Path:
    """Batch write helper (``features`` maps ``image_id -> [N, D]``)."""
    path = Path(path)
    for bank in banks:
        write_bank(
            path,
            bank,
            None if features is None else features.get(int(bank.image_id)),
            mode=mode,
            backend=backend,
        )
        mode = "a"  # first call creates/truncates, the rest append
    return path


def read_bank(
    path: str | Path,
    image_id: Optional[int] = None,
    *,
    backend: Optional[str] = None,
) -> ProposalBank:
    """Read a single bank; ``image_id=None`` returns the first stored image.

    Crop embeddings are *not* loaded here (they would defeat the whole point of
    the cache): the returned bank carries ``feature_ref`` when the cache holds a
    ``features`` dataset for that image, and callers pull rows with
    :func:`read_features`.
    """
    chosen = _detect_backend(path, backend)
    if chosen == "hdf5":
        return _h5_read(path, image_id)
    return _npz_read(path, image_id)


def iter_banks(path: str | Path, *, backend: Optional[str] = None) -> Iterator[ProposalBank]:
    """Batch traversal API used by the audits and the candidate builder."""
    chosen = _detect_backend(path, backend)
    for image_id in list_image_ids(path, backend=chosen):
        yield read_bank(path, image_id, backend=chosen)


def read_all_banks(path: str | Path, *, backend: Optional[str] = None) -> Dict[int, ProposalBank]:
    return {int(bank.image_id): bank for bank in iter_banks(path, backend=backend)}


def list_image_ids(path: str | Path, *, backend: Optional[str] = None) -> List[int]:
    chosen = _detect_backend(path, backend)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"proposal cache {path} not found - feature extraction is a Phase 0 checklist step"
        )
    if chosen == "hdf5":
        import h5py

        with h5py.File(path, "r") as handle:
            group = handle.get("images", handle)
            return sorted(int(key) for key in group.keys())
    with np.load(path, allow_pickle=False) as store:
        image_ids = {int(key.split("::")[0]) for key in store.files if "::" in key}
    return sorted(image_ids)


def read_features(path: str | Path, image_id: int, *, backend: Optional[str] = None) -> np.ndarray:
    """Lazy read of the cached ``[N, D]`` crop embeddings (float32 on return)."""
    chosen = _detect_backend(path, backend)
    if chosen == "hdf5":
        import h5py

        with h5py.File(path, "r") as handle:
            node = handle[f"images/{int(image_id)}/features"]
            return np.asarray(node[:], dtype=np.float32)
    with np.load(path, allow_pickle=False) as store:
        key = f"{int(image_id)}::features"
        if key not in store:
            raise KeyError(f"{path} has no cached features for image {image_id}")
        return np.asarray(store[key], dtype=np.float32)


# --- HDF5 backend ----------------------------------------------------------
def _h5_write(path: Path, bank: ProposalBank, features: Optional[np.ndarray], mode: str) -> None:
    import h5py

    h5_mode = {"a": "a", "w": "w", "x": "x", "r+": "r+"}[mode]
    with h5py.File(path, h5_mode) as handle:
        images = handle.require_group("images")
        group = images.require_group(str(int(bank.image_id)))
        for name, array in _bank_payload(bank).items():
            if name in group:
                del group[name]
            group.create_dataset(name, data=array, compression="gzip")
        if features is not None:
            if "features" in group:
                del group["features"]
            group.create_dataset(
                "features", data=np.asarray(features, dtype=np.float16), compression="gzip"
            )
        group.attrs["feature_ref"] = feature_ref(path, bank.image_id)
        handle.attrs["backend"] = "hdf5"


def _h5_read(path: Path, image_id: Optional[int]) -> ProposalBank:
    import h5py

    with h5py.File(path, "r") as handle:
        images = handle.get("images", handle)
        keys = sorted(images.keys(), key=lambda k: int(k))
        if not keys:
            raise ValueError(f"{path} contains no proposal banks")
        target = int(keys[0]) if image_id is None else int(image_id)
        name = str(target)
        if name not in images:
            raise KeyError(f"image {target} not in cache {path}")
        node = images[name]
        stored = {key: np.asarray(node[key]) for key in node.keys()}
        ref = str(node.attrs["feature_ref"]) if "feature_ref" in node.attrs else feature_ref(path, target)
    return _bank_from_payload(stored, target, ref, path)


# --- NPZ backend ----------------------------------------------------------
def _npz_write(path: Path, bank: ProposalBank, features: Optional[np.ndarray], mode: str) -> None:
    payload = {f"{int(bank.image_id)}::{name}": array for name, array in _bank_payload(bank).items()}
    if features is not None:
        payload[f"{int(bank.image_id)}::features"] = np.asarray(features, dtype=np.float16)
    existing: Dict[str, np.ndarray] = {}
    if mode != "w" and path.exists():
        with np.load(path, allow_pickle=False) as store:
            existing = {key: store[key] for key in store.files}
    existing.update(payload)
    np.savez_compressed(path, **existing)


def _npz_read(path: Path, image_id: Optional[int]) -> ProposalBank:
    with np.load(path, allow_pickle=False) as store:
        available = sorted({int(key.split("::")[0]) for key in store.files if "::" in key})
        if not available:
            raise ValueError(f"{path} contains no proposal banks")
        target = available[0] if image_id is None else int(image_id)
        if target not in available:
            raise KeyError(f"image {target} not in cache {path}")
        stored = {
            key.split("::")[1]: store[key] for key in store.files if key.startswith(f"{target}::")
        }
    return _bank_from_payload(stored, target, feature_ref(path, target), path)


def _bank_from_payload(
    stored: Dict[str, np.ndarray], image_id: int, ref: Optional[str], path: Path
) -> ProposalBank:
    """Rebuild a bank from cache arrays (feature arrays stay on disk)."""
    if "boxes" not in stored or "objectness" not in stored:
        raise KeyError(f"cache {path} entry {image_id} lacks boxes/objectness")
    return ProposalBank(
        image_id=int(image_id),
        boxes=stored["boxes"],
        objectness=stored["objectness"],
        gt_ious=stored.get("gt_iou"),
        gt_assignment=stored.get("gt_assignment"),
        feature_ref=ref if "features" in stored else None,
    )


def dump_audit_json(report: Mapping[str, object], path: str | Path) -> Path:
    """Small helper so audits can be committed as artefacts (never in-memory only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=float)
    return path
