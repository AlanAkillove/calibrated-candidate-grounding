"""Split names and image-level validation splitting (protocol section 9).

The RefCOCO+ *validation* split must be cut **by image** into two disjoint
purposes:

``val_select``
    architecture / optimizer / early-stopping / hyper-parameter selection.

``val_calib``
    temperature, thresholds, calibration models, abstention thresholds *only*.

Cutting by image (never by expression) is mandatory: expressions that share an
image share proposals, crops and CLIP embeddings, so an expression-level cut
would leak calibration signal into model selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .types import ReferringExample

__all__ = [
    "SPLIT_TRAIN",
    "SPLIT_VAL",
    "SPLIT_VAL_SELECT",
    "SPLIT_VAL_CALIB",
    "SPLIT_TEST",
    "SPLIT_TEST_A",
    "SPLIT_TEST_B",
    "SPLIT_NATURAL_OMISSION",
    "SPLIT_NAMES",
    "TEST_SPLITS",
    "SELECTION_SPLITS",
    "CALIBRATION_SPLITS",
    "TRAIN_SPLITS",
    "DEFAULT_SPLIT_SEED",
    "SplitPlan",
    "normalize_split_name",
    "plan_val_split",
    "split_val_images",
    "assign_val_subsplits",
    "apply_split_plan",
    "split_examples",
    "image_ids_of",
    "find_image_leakage",
]

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_VAL_SELECT = "val_select"
SPLIT_VAL_CALIB = "val_calib"
SPLIT_TEST = "test"
SPLIT_TEST_A = "testA"
SPLIT_TEST_B = "testB"
#: Evaluation split holding expressions whose target is naturally missing from
#: the proposal bank (``max_i IoU(p_i, b*) < iou_threshold``).  Kept separate
#: from synthetic omission (protocol section 6).
SPLIT_NATURAL_OMISSION = "natural_omission"

SPLIT_NAMES: tuple[str, ...] = (
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLIT_VAL_SELECT,
    SPLIT_VAL_CALIB,
    SPLIT_TEST,
    SPLIT_TEST_A,
    SPLIT_TEST_B,
    SPLIT_NATURAL_OMISSION,
)

TRAIN_SPLITS: tuple[str, ...] = (SPLIT_TRAIN,)
SELECTION_SPLITS: tuple[str, ...] = (SPLIT_VAL_SELECT,)
CALIBRATION_SPLITS: tuple[str, ...] = (SPLIT_VAL_CALIB,)
TEST_SPLITS: tuple[str, ...] = (SPLIT_TEST, SPLIT_TEST_A, SPLIT_TEST_B)

#: Fixed seed for the val -> val_select / val_calib cut.  Changing it is a
#: protocol amendment, not an experiment knob.
DEFAULT_SPLIT_SEED = 0
DEFAULT_CALIB_FRACTION = 0.5

# Aliases encountered in the wild (UNC mats / matlab cell arrays).
_SPLIT_ALIASES: Mapping[str, str] = {
    "training": SPLIT_TRAIN,
    "tr": SPLIT_TRAIN,
    "validation": SPLIT_VAL,
    "valid": SPLIT_VAL,
    "val": SPLIT_VAL,
    "val_unc": SPLIT_VAL,
    "test": SPLIT_TEST,
    "testing": SPLIT_TEST,
    "testa": SPLIT_TEST_A,
    "testb": SPLIT_TEST_B,
    "test_a": SPLIT_TEST_A,
    "test_b": SPLIT_TEST_B,
    "testu": SPLIT_VAL,  # RefCOCOg "test-u" is a validation-style split
    "test_u": SPLIT_VAL,
    "val_u": SPLIT_VAL,
}


def normalize_split_name(name: object) -> str:
    """Map a raw annotation split label onto a canonical :data:`SPLIT_NAMES` entry.

    Unknown labels are lower-cased and returned as-is (they are still valid
    ``ReferringExample.split`` strings); this keeps the parser from silently
    dropping examples whose label spelling differs.  Whether an unknown label
    appeared is audited by :func:`ccg.data.refcoco`'s report, not here.
    """
    if name is None:
        return ""
    raw = str(name).strip()
    if not raw:
        return ""
    key = raw.lower().replace("-", "_")
    if key in _SPLIT_ALIASES:
        return _SPLIT_ALIASES[key]
    if raw in SPLIT_NAMES:
        return raw
    return key


@dataclass
class SplitPlan:
    """Deterministic description of the val -> (val_select, val_calib) cut."""

    val_select_image_ids: np.ndarray
    val_calib_image_ids: np.ndarray
    seed: int = DEFAULT_SPLIT_SEED
    calib_fraction: float = DEFAULT_CALIB_FRACTION

    def __post_init__(self) -> None:
        self.val_select_image_ids = np.unique(
            np.asarray(self.val_select_image_ids, dtype=np.int64).reshape(-1)
        )
        self.val_calib_image_ids = np.unique(
            np.asarray(self.val_calib_image_ids, dtype=np.int64).reshape(-1)
        )
        overlap = set(self.val_select_image_ids.tolist()) & set(self.val_calib_image_ids.tolist())
        if overlap:
            raise ValueError(
                f"val_select and val_calib share {len(overlap)} image ids: {sorted(overlap)[:5]}"
            )

    @property
    def num_images(self) -> int:
        return int(self.val_select_image_ids.size + self.val_calib_image_ids.size)

    def image_to_subsplit(self) -> Dict[int, str]:
        out: Dict[int, str] = {}
        for image_id in self.val_select_image_ids.tolist():
            out[int(image_id)] = SPLIT_VAL_SELECT
        for image_id in self.val_calib_image_ids.tolist():
            out[int(image_id)] = SPLIT_VAL_CALIB
        return out

    def to_dict(self) -> dict:
        return {
            "seed": int(self.seed),
            "calib_fraction": float(self.calib_fraction),
            "num_val_images": self.num_images,
            "val_select_image_ids": self.val_select_image_ids.tolist(),
            "val_calib_image_ids": self.val_calib_image_ids.tolist(),
        }


def split_val_images(
    val_image_ids: Iterable[int],
    seed: int = DEFAULT_SPLIT_SEED,
    calib_fraction: float = DEFAULT_CALIB_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Split *image ids* of the validation split into ``(val_select, val_calib)``.

    Deterministic given ``seed``: unique ids are sorted, then a fixed
    ``numpy.random.Generator`` permutation picks ``calib_fraction`` of them for
    calibration.  Because the unit of the cut is the image, every expression on
    a given image lands in the same sub-split by construction.

    Raises
    ------
    ValueError
        When fewer than two images are available (a 2-way cut is meaningless)
        or when ``calib_fraction`` is not in ``(0, 1)``.
    """
    if not 0.0 < float(calib_fraction) < 1.0:
        raise ValueError(f"calib_fraction must be in (0,1), got {calib_fraction}")
    images = np.unique(np.asarray(list(val_image_ids), dtype=np.int64).reshape(-1))
    if images.size < 2:
        raise ValueError(
            f"need at least 2 validation images to cut val_select/val_calib, got {images.size}"
        )
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(images)
    n_calib = int(round(order.size * float(calib_fraction)))
    n_calib = max(1, min(n_calib, order.size - 1))
    val_calib = np.sort(order[:n_calib])
    val_select = np.sort(order[n_calib:])
    return val_select, val_calib


def plan_val_split(
    examples: Sequence[ReferringExample],
    seed: int = DEFAULT_SPLIT_SEED,
    calib_fraction: float = DEFAULT_CALIB_FRACTION,
) -> SplitPlan:
    """Build a :class:`SplitPlan` from the ``val`` images found in ``examples``."""
    val_images = [ex.image_id for ex in examples if normalize_split_name(ex.split) == SPLIT_VAL]
    if not val_images:
        raise ValueError("no examples with split='val' found; cannot plan val_select/val_calib")
    val_select, val_calib = split_val_images(val_images, seed=seed, calib_fraction=calib_fraction)
    return SplitPlan(
        val_select_image_ids=val_select,
        val_calib_image_ids=val_calib,
        seed=int(seed),
        calib_fraction=float(calib_fraction),
    )


def assign_val_subsplits(
    image_ids: Iterable[int],
    val_select_image_ids: Iterable[int],
    val_calib_image_ids: Iterable[int],
) -> np.ndarray:
    """Label each image id with ``val_select`` / ``val_calib``.

    Anything that is not in either list raises: silently labelling unknown
    images would let test data drift into the calibration split.
    """
    lookup = {
        int(image_id): SPLIT_VAL_SELECT for image_id in np.asarray(val_select_image_ids).reshape(-1)
    }
    lookup.update(
        {int(image_id): SPLIT_VAL_CALIB for image_id in np.asarray(val_calib_image_ids).reshape(-1)}
    )
    out = np.empty(0, dtype="<U16")
    labels: List[str] = []
    for image_id in image_ids:
        label = lookup.get(int(image_id))
        if label is None:
            raise ValueError(f"image_id {int(image_id)} belongs to neither val_select nor val_calib")
        labels.append(label)
    return np.asarray(labels, dtype="<U16")


def apply_split_plan(
    examples: Sequence[ReferringExample], plan: SplitPlan
) -> Dict[str, List[ReferringExample]]:
    """Route examples into ``train`` / ``val_select`` / ``val_calib`` / ``test*``.

    Returns a dict keyed by canonical split name; empty keys are omitted so
    that a missing ``testB`` is visible instead of masquerading as size 0.
    ``val`` examples are rewritten in place (a shallow copy with a new ``split``
    field) so downstream code never sees the ambiguous ``"val"`` label.
    """
    lookup = plan.image_to_subsplit()
    out: Dict[str, List[ReferringExample]] = {}
    for ex in examples:
        name = normalize_split_name(ex.split)
        if name == SPLIT_VAL:
            name = lookup.get(int(ex.image_id))
            if name is None:
                raise ValueError(
                    f"example {ex.ref_id}: image {ex.image_id} is not covered by the val split plan"
                )
            ex = _relabel(ex, name)
        elif name == "":
            raise ValueError(f"example {ex.ref_id} has an empty split label")
        out.setdefault(name, []).append(ex)
    return out


def split_examples(
    examples: Sequence[ReferringExample],
    seed: int = DEFAULT_SPLIT_SEED,
    calib_fraction: float = DEFAULT_CALIB_FRACTION,
) -> Dict[str, List[ReferringExample]]:
    """Convenience wrapper: :func:`plan_val_split` + :func:`apply_split_plan`."""
    return apply_split_plan(examples, plan_val_split(examples, seed=seed, calib_fraction=calib_fraction))


def image_ids_of(examples: Iterable[ReferringExample]) -> np.ndarray:
    """Sorted unique image ids of a group of examples."""
    return np.unique(np.asarray([ex.image_id for ex in examples], dtype=np.int64))


def find_image_leakage(groups: Mapping[str, Sequence[ReferringExample]]) -> List[str]:
    """Report image ids appearing in more than one group.

    Used as a sanity check before any experiment: an empty list means the
    groups are image-disjoint.  ``val_select`` vs ``val_calib`` must always
    return ``[]``.
    """
    names = list(groups.keys())
    seen: Dict[int, str] = {}
    violations: List[str] = []
    for name in names:
        for image_id in image_ids_of(groups[name]).tolist():
            previous = seen.get(int(image_id))
            if previous is not None:
                violations.append(f"{previous} <-> {name} share image {int(image_id)}")
            else:
                seen[int(image_id)] = name
    return violations


def _relabel(example: ReferringExample, split: str) -> ReferringExample:
    """Copy a :class:`ReferringExample` with a new ``split`` label."""
    return ReferringExample(
        ref_id=example.ref_id,
        image_id=example.image_id,
        text=example.text,
        gt_box=example.gt_box,
        gt_object_id=example.gt_object_id,
        split=split,
    )
