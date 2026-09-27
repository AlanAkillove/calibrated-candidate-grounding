"""Core data structures for calibrated candidate grounding (Task 4).

These are *pure data containers*: numpy arrays plus dataclasses.  They must not
import torch, OpenCLIP, h5py or any model / metric code, because the same
structures are shared by the offline feature cache, the candidate-set builder,
the decision models and the evaluation scripts.

Design rules (from the project protocol, sections 4-7):

* A :class:`ProposalBank` is query-independent: one fixed set of proposals per
  image, reused by every referring expression on that image.
* Proposal *features* (CLIP crop embeddings) are NOT stored as in-memory
  dataclass fields.  They live in the on-disk feature cache and a bank only
  carries ``feature_ref`` (a string handle such as
  ``cache://proposals.h5#/images/391893/features``).
* A :class:`CandidateSet` references rows of a :class:`ProposalBank` through
  ``candidate_indices`` and knows *where* the target sits inside the set
  (``target_index``), or that the target is absent (``target_index is None``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "HARDNESS_VALUES",
    "CandidateSet",
    "ProposalBank",
    "ReferringExample",
    "box_area",
    "box_xywh",
]

#: Candidate-set hardness levels supported in Phase 0 (section 5 of the protocol).
HARDNESS_VALUES: tuple[str, ...] = ("random", "same_category_hard", "clip_hard")

_BOX_DTYPE = np.float32
_INDEX_DTYPE = np.int64


def _as_1d(values, name: str, dtype, *, allow_none: bool = False):
    """Coerce ``values`` to a 1-D array of ``dtype`` (or ``None``)."""
    if values is None:
        if allow_none:
            return None
        raise ValueError(f"{name} must not be None")
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=dtype)


def _as_box(box, name: str) -> np.ndarray:
    """Coerce ``box`` to a finite, non-degenerate ``xyxy`` float32 vector."""
    arr = np.asarray(box, dtype=_BOX_DTYPE)
    if arr.shape != (4,):
        raise ValueError(f"{name} must have shape (4,) as xyxy, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values: {arr.tolist()}")
    x1, y1, x2, y2 = arr.tolist()
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"{name} must satisfy x2>x1 and y2>y1 (xyxy), got {arr.tolist()}"
        )
    return arr


@dataclass(eq=False)
class ReferringExample:
    """One referring expression (a ``referring instance`` in RefCOCO terms).

    Attributes
    ----------
    ref_id:
        Unique id of the expression inside its dataset.
    image_id:
        COCO image id the expression points to.
    text:
        The natural-language expression.
    gt_box:
        Ground-truth box ``[x1, y1, x2, y2]`` in image pixel coordinates.
    gt_object_id:
        Id of the COCO GT object the box belongs to, when the annotation format
        provides it.  ``None`` otherwise (RefCOCO+ json does not, UNC mats may).
    split:
        Split name, e.g. ``"train"``, ``"val"``, ``"testA"``, or a project
        sub-split such as ``"val_calib"`` (see :mod:`ccg.data.splits`).
    """

    ref_id: int
    image_id: int
    text: str
    gt_box: np.ndarray
    gt_object_id: Optional[int] = None
    split: str = ""

    def __post_init__(self) -> None:
        self.ref_id = int(self.ref_id)
        self.image_id = int(self.image_id)
        if self.ref_id < 0:
            raise ValueError(f"ref_id must be non-negative, got {self.ref_id}")
        if self.image_id < 0:
            raise ValueError(f"image_id must be non-negative, got {self.image_id}")
        if not isinstance(self.text, str):
            raise ValueError(f"text must be a str, got {type(self.text)}")
        if not self.text.strip():
            raise ValueError(f"example {self.ref_id}: text must be a non-empty string")
        self.gt_box = _as_box(self.gt_box, "gt_box")
        if self.gt_object_id is not None:
            self.gt_object_id = int(self.gt_object_id)
        if self.split is None:
            raise ValueError("split must be a str (use '' when unknown)")
        if not isinstance(self.split, str):
            raise ValueError(f"split must be a str, got {type(self.split)}")
        if any(ch.isspace() for ch in self.split):
            raise ValueError(f"split must not contain whitespace, got {self.split!r}")

    # -- convenience ---------------------------------------------------------
    @property
    def area(self) -> float:
        """Pixel area of the ground-truth box."""
        x1, y1, x2, y2 = self.gt_box
        return float((x2 - x1) * (y2 - y1))

    def __repr__(self) -> str:  # keep repr compact and array-free of truncation
        return (
            f"ReferringExample(ref_id={self.ref_id}, image_id={self.image_id}, "
            f"split={self.split!r}, gt_box={self.gt_box.tolist()}, "
            f"text={self.text[:40]!r}{'...' if len(self.text) > 40 else ''})"
        )


@dataclass(eq=False)
class ProposalBank:
    """Fixed, query-independent proposal set for one image (section 4).

    ``boxes`` are rows of the bank; every index used by a
    :class:`CandidateSet` refers to a row of ``boxes``.

    Attributes
    ----------
    image_id:
        COCO image id.
    boxes:
        ``[N, 4]`` float32 ``xyxy`` proposal boxes.
    objectness:
        ``[N]`` float32 detector objectness / score.
    gt_ious:
        ``[N]`` float32 IoU between each proposal and its *nearest* GT object
        (``None`` when the image has no GT annotation available).
    gt_assignment:
        ``[N]`` int64 id of the nearest GT object, ``-1`` when unassigned.
    feature_ref:
        Handle to the on-disk ``[N, D]`` crop-embedding cache.  Features are
        deliberately *not* an in-memory field (protocol section 7): changing the
        candidate sampling protocol must never require re-extracting CLIP
        features, and keeping the array out of the dataclass avoids accidentally
        pinning gigabytes in RAM.
    """

    image_id: int
    boxes: np.ndarray
    objectness: np.ndarray
    gt_ious: Optional[np.ndarray] = None
    gt_assignment: Optional[np.ndarray] = None
    feature_ref: Optional[str] = None

    def __post_init__(self) -> None:
        self.image_id = int(self.image_id)
        if self.image_id < 0:
            raise ValueError(f"image_id must be non-negative, got {self.image_id}")

        boxes = np.asarray(self.boxes, dtype=_BOX_DTYPE)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"boxes must have shape [N,4], got {boxes.shape}")
        if boxes.shape[0] == 0:
            raise ValueError("boxes must contain at least one proposal")
        if not np.all(np.isfinite(boxes)):
            raise ValueError("boxes contain non-finite values")
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        if np.any(widths < 0) or np.any(heights < 0):
            bad = int(np.argmax((widths < 0) | (heights < 0)))
            raise ValueError(
                f"boxes must be xyxy with x2>=x1 and y2>=y1; row {bad} "
                f"= {boxes[bad].tolist()}"
            )
        self.boxes = boxes

        n = boxes.shape[0]
        self.objectness = _as_1d(self.objectness, "objectness", _BOX_DTYPE)
        if self.objectness.shape[0] != n:
            raise ValueError(
                f"objectness has length {self.objectness.shape[0]}, expected {n}"
            )

        if self.gt_ious is not None:
            self.gt_ious = _as_1d(self.gt_ious, "gt_ious", _BOX_DTYPE)
            if self.gt_ious.shape[0] != n:
                raise ValueError(f"gt_ious must have length {n}")
            if np.any(self.gt_ious < 0) or np.any(self.gt_ious > 1):
                raise ValueError("gt_ious values must lie in [0, 1]")
        if self.gt_assignment is not None:
            self.gt_assignment = _as_1d(self.gt_assignment, "gt_assignment", _INDEX_DTYPE)
            if self.gt_assignment.shape[0] != n:
                raise ValueError(f"gt_assignment must have length {n}")
            if np.any(self.gt_assignment < -1):
                raise ValueError("gt_assignment uses -1 for 'unassigned'; got < -1")
        if (self.gt_ious is None) != (self.gt_assignment is None):
            raise ValueError("gt_ious and gt_assignment must be provided together")

        if self.feature_ref is not None and not isinstance(self.feature_ref, str):
            raise ValueError("feature_ref must be a str or None")

    # -- convenience ---------------------------------------------------------
    @property
    def N(self) -> int:
        """Number of proposals in the bank."""
        return int(self.boxes.shape[0])

    def top_k_order(self, k: Optional[int] = None) -> np.ndarray:
        """Row indices of the ``k`` most confident proposals (descending)."""
        order = np.argsort(-self.objectness, kind="stable")
        return order if k is None else order[: int(k)]

    def subset(self, indices: Sequence[int] | np.ndarray) -> "ProposalBank":
        """Return a new bank restricted to ``indices`` (row order preserved)."""
        idx = _as_1d(indices, "indices", _INDEX_DTYPE)
        if idx.size == 0:
            raise ValueError("indices must be non-empty")
        if idx.min() < 0 or idx.max() >= self.N:
            raise ValueError(f"indices out of range [0,{self.N})")
        return ProposalBank(
            image_id=self.image_id,
            boxes=self.boxes[idx],
            objectness=self.objectness[idx],
            gt_ious=None if self.gt_ious is None else self.gt_ious[idx],
            gt_assignment=None if self.gt_assignment is None else self.gt_assignment[idx],
            feature_ref=None if self.feature_ref is None else f"{self.feature_ref}[{idx.tolist()}]",
        )

    def __repr__(self) -> str:
        return (
            f"ProposalBank(image_id={self.image_id}, N={self.N}, "
            f"features={'cached' if self.feature_ref else 'none'})"
        )


@dataclass(eq=False)
class CandidateSet:
    """A concrete K-way candidate set for one expression (section 5).

    Attributes
    ----------
    ref_id:
        Expression this candidate set belongs to.
    candidate_indices:
        ``[K]`` int64 row indices into the image's :class:`ProposalBank`.
    target_index:
        Position (0-based, *inside* ``candidate_indices``) of the target
        candidate, or ``None`` when the target is absent from the set.
    K:
        Set size.  ``None`` means "derive from ``len(candidate_indices)``";
        when given explicitly it must match.
    hardness:
        One of :data:`HARDNESS_VALUES`.
    regime:
        Free-form evaluation regime tag, e.g. ``"random"``,
        ``"clip_hard"``, ``"synthetic_omit"``, ``"natural_omission"``.
    target_present:
        ``True`` iff the target is in the set.  ``None`` means "derive from
        ``target_index is not None``"; an explicit value must be consistent.
    image_id:
        Optional denormalised image id, handy for image-level bootstrap.
    """

    ref_id: int
    candidate_indices: np.ndarray
    target_index: Optional[int] = None
    K: Optional[int] = None
    hardness: str = "random"
    regime: str = "random"
    target_present: Optional[bool] = None
    image_id: Optional[int] = None

    def __post_init__(self) -> None:
        self.ref_id = int(self.ref_id)
        idx = np.asarray(self.candidate_indices)
        if idx.ndim != 1:
            raise ValueError(
                f"candidate_indices must be a 1-D array of bank row indices, "
                f"got shape {idx.shape}"
            )
        idx = np.ascontiguousarray(idx, dtype=_INDEX_DTYPE)
        if idx.size == 0:
            raise ValueError("candidate_indices must contain at least one candidate")
        if np.any(idx < 0):
            raise ValueError("candidate_indices must be non-negative bank row ids")
        unique, counts = np.unique(idx, return_counts=True)
        duplicates = unique[counts > 1]
        if duplicates.size:
            raise ValueError(f"candidate_indices contains duplicates: {duplicates.tolist()}")
        self.candidate_indices = idx

        if self.K is None:
            self.K = int(idx.size)
        else:
            self.K = int(self.K)
            if self.K != int(idx.size):
                raise ValueError(f"K={self.K} inconsistent with len(candidate_indices)={idx.size}")

        if self.target_index is None:
            self.target_index = None
        else:
            self.target_index = int(self.target_index)
            if not 0 <= self.target_index < self.K:
                raise ValueError(
                    f"target_index={self.target_index} outside [0,{self.K}) for this candidate set"
                )

        derived_present = self.target_index is not None
        if self.target_present is None:
            self.target_present = derived_present
        else:
            self.target_present = bool(self.target_present)
            if self.target_present != derived_present:
                raise ValueError(
                    f"target_present={self.target_present} inconsistent with "
                    f"target_index={self.target_index} "
                    f"(derived target_present={derived_present})"
                )

        if self.hardness not in HARDNESS_VALUES:
            raise ValueError(
                f"unknown hardness {self.hardness!r}; expected one of {HARDNESS_VALUES}"
            )
        if not isinstance(self.regime, str) or not self.regime:
            raise ValueError(f"regime must be a non-empty str, got {self.regime!r}")
        if self.image_id is not None:
            self.image_id = int(self.image_id)

    # -- convenience ---------------------------------------------------------
    @property
    def target_proposal_index(self) -> Optional[int]:
        """Bank row id of the target candidate (``None`` when absent)."""
        if self.target_index is None:
            return None
        return int(self.candidate_indices[self.target_index])

    @property
    def distractor_indices(self) -> np.ndarray:
        """Bank row ids of every non-target candidate."""
        target = self.target_proposal_index
        if target is None:
            return self.candidate_indices.copy()
        return self.candidate_indices[self.candidate_indices != target]

    @property
    def contains_target(self) -> bool:
        return bool(self.target_present)

    def has_proposal(self, proposal_index: int) -> bool:
        return bool(np.any(self.candidate_indices == int(proposal_index)))

    def __repr__(self) -> str:
        return (
            f"CandidateSet(ref_id={self.ref_id}, K={self.K}, "
            f"target_index={self.target_index}, hardness={self.hardness!r}, "
            f"regime={self.regime!r}, target_present={self.target_present})"
        )


def box_xywh(boxes: np.ndarray) -> np.ndarray:
    """``xyxy`` -> ``[x, y, w, h]`` (used by annotation writers / audits)."""
    arr = np.asarray(boxes, dtype=_BOX_DTYPE)
    if arr.ndim == 1:
        arr = arr.reshape(1, 4)
    out = arr.copy()
    out[:, 2] = arr[:, 2] - arr[:, 0]
    out[:, 3] = arr[:, 3] - arr[:, 1]
    return out


def box_area(boxes: np.ndarray) -> np.ndarray:
    """Area of ``xyxy`` boxes."""
    arr = np.asarray(boxes, dtype=_BOX_DTYPE)
    if arr.ndim == 1:
        arr = arr.reshape(1, 4)
    return np.maximum(arr[:, 2] - arr[:, 0], 0.0) * np.maximum(arr[:, 3] - arr[:, 1], 0.0)
