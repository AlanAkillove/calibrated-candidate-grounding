"""Frozen candidate manifests: per-ref distractor orderings and the common cohort.

This module is the *single* place where the frozen candidate-set construction
rules are materialised (multi-agent brief, 2026-09-27).  One manifest row per
referring expression records

* the target's position inside the image's proposal bank (``None`` = natural
  miss - the expression stays in the file, never silently dropped),
* ONE frozen ordering of the valid distractors; the C_K candidate sets for
  every K are the *prefixes* ``C_K = target + order[:K-1]``, so the nesting
  ``C5 subset C10 subset C20 subset C50`` holds by construction and a single
  ordering serves every K,
* the eligibility of every primary K -
  ``eligible[K] = target present and n_valid_distractors >= K - 1``,
* the same-category bookkeeping used by the ``same_category`` regime,
* the identity of the cohort: :func:`common_cohort` returns exactly
  ``{entry : eligible[K] == True}`` for ``K = 50`` - because eligibility is
  monotone in K (``eligible[K] => eligible[K']`` for every ``K' <= K``), that
  set is the intersection of the per-K eligible sets, i.e. the primary Ks of
  protocol section 29 are evaluated on *identical ref_ids*.

Frozen ordering rules
---------------------
``random`` regime
    ``distractor_order`` is a deterministic shuffle of the valid distractor
    indices with ``rng = numpy.random.default_rng([seed, ref_id])`` -
    independent per ref and independent of the iteration order.  **One** shuffle
    per ref, shared by every K (per-K resampling is forbidden).

``same_category`` regime
    ``distractor_order = [same-category hard distractors (ascending bank
    index)] + [the same deterministic shuffle of the remaining valid
    distractors]``.  ``n_same_category_available`` records the actual supply;
    ``n_same_used_by_K = min(n_same_available, K-1)`` and
    ``hard_fraction_by_K = n_same_used / (K-1)`` describe the same-category
    share of the K-1 distractor prefix.  Both dicts are 0 / 0.0 under
    ``random`` (its ordering does not promote same-category distractors).

Valid distractor pool
    every bank index minus the target and the *target-equivalent* proposals
    (IoU >= 0.5 with the GT box, via :func:`ccg.data.proposals.assign_target`).
    The stored ordering is the full order, capped at ``top_n - 1`` entries
    (63 for the frozen ``N = 64`` bank); consumers take K-prefixes.

Same-category metadata
    a proposal inherits the category of its highest-IoU COCO GT object, but
    only when that IoU is ``>= 0.5``; otherwise it is ``-1`` ("unknown").
    Unknown categories never match - not even each other
    (:func:`ccg.data.audit.assign_gt_category` /
    :func:`ccg.data.audit.same_category_counts`).  Crowd objects are kept in
    the GT set (the audit's ``non_crowd`` view is a reporting choice, not a
    construction rule).

Storage
-------
:meth:`ManifestFile.save` writes one JSON line per entry plus a sibling
``{stem}.meta.json`` (``random_val.jsonl`` -> ``random_val.meta.json``).
Every entry of the ``val`` file additionally carries ``eval_split``
(``val_select`` / ``val_calib``), cut **by image** once with
``ccg.data.splits``' fixed ``DEFAULT_SPLIT_SEED`` - all expressions of an
image land in the same sub-split by construction.

Bank reader
-----------
The bank file is read per image directly from the frozen group layout
(``image_{image_id}`` with ``boxes [K,4] float32 xyxy`` and
``objectness [K] float32``) with a lazy h5py import, so manifests can be
built and unit tested without the parallel ``ccg.data.bank`` module having
landed; the layout is identical to that module's schema.  ``build_manifests``
validates existence itself and raises a clear error while the bank is not
built yet.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .audit import assign_gt_category, same_category_counts
from .coco import CocoIndex
from .coco import build_index as build_coco_index
from .proposals import DEFAULT_IOU_THRESH, assign_target, xywh_to_xyxy
from .refcoco import load_instances_json, load_refs_pickle
from .splits import (
    DEFAULT_SPLIT_SEED,
    SPLIT_TEST_A,
    SPLIT_TEST_B,
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLIT_VAL_CALIB,
    SPLIT_VAL_SELECT,
    normalize_split_name,
)
#: The frozen multi-agent brief calls the image-level cut
#: ``split_val_by_image``; :mod:`ccg.data.splits` exports it as
#: :func:`split_val_images`.  Bind the brief's spelling as a local alias.
from .splits import split_val_images as split_val_by_image
from .types import ProposalBank

__all__ = [
    "MANIFEST_SEED",
    "PRIMARY_KS",
    "SCHEMA_VERSION",
    "REGIMES",
    "FILE_SPLITS",
    "ManifestEntry",
    "ManifestFile",
    "build_manifests",
    "common_cohort",
    "filter_entries",
    "manifest_path",
]

#: Fixed seed of the manifest construction (protocol amendment, 2026-09-27).
MANIFEST_SEED = 20260927
#: The primary candidate-set sizes evaluated in Phase 0.
PRIMARY_KS: Tuple[int, ...] = (5, 10, 20, 50)
#: Version tag written into every meta file (checked on load).
SCHEMA_VERSION = "manifests-v1"
#: Construction regimes a manifest file can be built for.
REGIMES: Tuple[str, ...] = ("random", "same_category")
#: The four UNC split files a build is materialised into.
FILE_SPLITS: Tuple[str, ...] = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST_A, SPLIT_TEST_B)

_BANK_GROUP_PREFIX = "image_"
_FINGERPRINT_PREFIX_BYTES = 8 * 1024 * 1024
_UNKNOWN_CATEGORY = -1


# ---------------------------------------------------------------------------
# results containers
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class ManifestEntry:
    """One frozen row: target position, distractor ordering, eligibility.

    Attributes
    ----------
    ref_id / image_id / split:
        Identity of the expression; ``split`` is the canonicalised UNC split
        (``train`` / ``val`` / ``testA`` / ``testB``).
    target_index:
        Bank row of the unique target (highest IoU >= 0.5), or ``None`` for a
        natural miss.
    distractor_order:
        ``int32`` frozen ordering of the valid distractor indices (the full
        order capped at ``top_n - 1`` entries); ``C_K = target +
        order[:K-1]`` for every primary K.
    n_valid_distractors:
        Size of the valid pool (bank rows minus target minus target-equivalent
        proposals), measured *before* the storage cap.
    n_target_equiv_removed:
        Number of other proposals with IoU >= 0.5 (removed as second correct
        answers).
    n_same_category_available:
        Valid distractors sharing the target's proposal category (``0`` when
        the target's category is unknown / no GT metadata was supplied).
    eligible:
        ``K -> (target present and n_valid_distractors >= K - 1)``.
    n_same_used_by_K:
        ``K -> min(n_same_category_available, K-1)`` (``0`` under
        ``random``).
    hard_fraction_by_K:
        ``K -> n_same_used / (K-1)`` (``0.0`` under ``random``).
    target_max_iou:
        ``max_i IoU(p_i, b*)`` of the expression (``< 0.5`` for a miss).
    eval_split:
        ``val_select`` / ``val_calib`` on ``val`` entries (by image, fixed
        seed), ``None`` everywhere else.
    """

    ref_id: int
    image_id: int
    split: str
    target_index: Optional[int]
    distractor_order: np.ndarray
    n_valid_distractors: int
    n_target_equiv_removed: int
    n_same_category_available: int
    eligible: Dict[int, bool]
    n_same_used_by_K: Dict[int, int]
    hard_fraction_by_K: Dict[int, float]
    target_max_iou: float
    eval_split: Optional[str] = None

    def __post_init__(self) -> None:
        self.ref_id = int(self.ref_id)
        self.image_id = int(self.image_id)
        self.split = str(self.split)
        self.target_index = None if self.target_index is None else int(self.target_index)

        order = np.asarray(self.distractor_order)
        if order.ndim != 1:
            raise ValueError(f"distractor_order must be 1-D, got shape {order.shape}")
        order = np.ascontiguousarray(order, dtype=np.int32)
        if order.size:
            if int(order.min()) < 0:
                raise ValueError("distractor_order must hold non-negative bank indices")
            if np.unique(order).size != order.size:
                raise ValueError("distractor_order contains duplicate indices")
            if self.target_index is not None and bool(np.any(order == self.target_index)):
                raise ValueError(
                    f"distractor_order must not contain the target (ref {self.ref_id})"
                )
        self.distractor_order = order

        self.n_valid_distractors = int(self.n_valid_distractors)
        self.n_target_equiv_removed = int(self.n_target_equiv_removed)
        self.n_same_category_available = int(self.n_same_category_available)
        if self.n_valid_distractors < order.size:
            raise ValueError(
                f"n_valid_distractors ({self.n_valid_distractors}) is smaller than the "
                f"stored ordering ({order.size}) for ref {self.ref_id}"
            )
        self.eligible = {int(key): bool(value) for key, value in dict(self.eligible).items()}
        self.n_same_used_by_K = {
            int(key): int(value) for key, value in dict(self.n_same_used_by_K).items()
        }
        self.hard_fraction_by_K = {
            int(key): float(value) for key, value in dict(self.hard_fraction_by_K).items()
        }
        self.target_max_iou = float(self.target_max_iou)
        if self.eval_split is not None:
            self.eval_split = str(self.eval_split)

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict (arrays as lists, int-keyed dicts as str keys)."""
        payload: Dict[str, Any] = {
            "ref_id": self.ref_id,
            "image_id": self.image_id,
            "split": self.split,
            "target_index": self.target_index,
            "distractor_order": self.distractor_order.astype(int).tolist(),
            "n_valid_distractors": self.n_valid_distractors,
            "n_target_equiv_removed": self.n_target_equiv_removed,
            "n_same_category_available": self.n_same_category_available,
            "eligible": {str(k): bool(v) for k, v in sorted(self.eligible.items())},
            "n_same_used_by_K": {
                str(k): int(v) for k, v in sorted(self.n_same_used_by_K.items())
            },
            "hard_fraction_by_K": {
                str(k): float(v) for k, v in sorted(self.hard_fraction_by_K.items())
            },
            "target_max_iou": self.target_max_iou,
        }
        if self.eval_split is not None:
            payload["eval_split"] = self.eval_split
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ManifestEntry":
        """Rebuild an entry from :meth:`to_dict` output (str keys -> int)."""
        def _int_keyed(source: Mapping[Any, Any]) -> Dict[int, Any]:
            return {int(key): value for key, value in dict(source).items()}

        return cls(
            ref_id=int(payload["ref_id"]),
            image_id=int(payload["image_id"]),
            split=str(payload["split"]),
            target_index=(
                None if payload.get("target_index") is None else int(payload["target_index"])
            ),
            distractor_order=np.asarray(payload["distractor_order"], dtype=np.int32),
            n_valid_distractors=int(payload["n_valid_distractors"]),
            n_target_equiv_removed=int(payload["n_target_equiv_removed"]),
            n_same_category_available=int(payload["n_same_category_available"]),
            eligible={k: bool(v) for k, v in _int_keyed(payload["eligible"]).items()},
            n_same_used_by_K={
                k: int(v) for k, v in _int_keyed(payload["n_same_used_by_K"]).items()
            },
            hard_fraction_by_K={
                k: float(v) for k, v in _int_keyed(payload["hard_fraction_by_K"]).items()
            },
            target_max_iou=float(payload["target_max_iou"]),
            eval_split=payload.get("eval_split"),
        )


@dataclass(eq=False)
class ManifestFile:
    """A manifest for one construction regime, saved as jsonl + meta json."""

    regime: str
    meta: Dict[str, Any]
    entries: List[ManifestEntry]

    def __post_init__(self) -> None:
        self.regime = str(self.regime)
        if not self.regime:
            raise ValueError("regime must be a non-empty string")
        self.meta = dict(self.meta)
        self.entries = list(self.entries)
        for position, entry in enumerate(self.entries):
            if not isinstance(entry, ManifestEntry):
                raise TypeError(f"entries[{position}] is not a ManifestEntry: {type(entry)}")

    # -- persistence ---------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Write ``path`` (jsonl, one entry per line) + ``path.with_suffix('.meta.json')``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for entry in self.entries:
                handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        path.with_suffix(".meta.json").write_text(
            json.dumps(self.meta, indent=2, ensure_ascii=False, default=_json_fallback) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ManifestFile":
        """Read back a :meth:`save` pair (jsonl + sibling ``.meta.json``)."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"manifest file {path} not found")
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"manifest meta file {meta_path} not found - a manifest is always the "
                "jsonl + meta pair written by ManifestFile.save()"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, Mapping):
            raise ValueError(f"{meta_path}: meta must be a JSON object")
        version = meta.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{meta_path}: schema_version {version!r} does not match {SCHEMA_VERSION!r}"
            )
        entries: List[ManifestEntry] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(ManifestEntry.from_dict(json.loads(line)))
                except Exception as exc:
                    raise ValueError(f"{path}:{line_no}: malformed manifest entry: {exc}") from exc
        return cls(regime=str(meta.get("regime", "")), meta=dict(meta), entries=entries)

    # -- helpers -------------------------------------------------------------
    def subset(self, entries: Sequence[ManifestEntry]) -> "ManifestFile":
        """A new file over ``entries`` with the size statistics recomputed."""
        subset = list(entries)
        meta = dict(self.meta)
        meta["n_refs"] = len(subset)
        meta["n_images"] = len({int(entry.image_id) for entry in subset})
        counts: Dict[str, int] = {}
        for entry in subset:
            counts[entry.split] = counts.get(entry.split, 0) + 1
        meta["split_counts"] = {key: counts[key] for key in sorted(counts)}
        return ManifestFile(regime=self.regime, meta=meta, entries=subset)


# ---------------------------------------------------------------------------
# module-level helpers (frozen interface)
# ---------------------------------------------------------------------------
def filter_entries(
    manifest: Union[ManifestFile, Iterable[ManifestEntry]],
    split: Optional[str] = None,
) -> List[ManifestEntry]:
    """Entries of ``manifest`` selected by ``split``.

    ``None`` returns every entry.  ``train`` / ``val`` / ``testA`` / ``testB``
    match the UNC split of the entry (``val`` keeps its file's ``eval_split``
    column untouched); ``val_select`` / ``val_calib`` match the entry's
    ``eval_split`` (only present on ``val`` entries).  Anything else raises -
    a typo must not silently return an empty cohort.
    """
    entries = manifest.entries if isinstance(manifest, ManifestFile) else list(manifest)
    if split is None:
        return list(entries)
    name = normalize_split_name(split)
    if name in FILE_SPLITS:
        return [entry for entry in entries if entry.split == name]
    if name in (SPLIT_VAL_SELECT, SPLIT_VAL_CALIB):
        return [entry for entry in entries if entry.eval_split == name]
    raise ValueError(
        f"unknown split {split!r}; expected one of "
        f"{FILE_SPLITS + (SPLIT_VAL_SELECT, SPLIT_VAL_CALIB)}"
    )


def common_cohort(
    entries: Union[ManifestFile, Iterable[ManifestEntry]], K: int = 50
) -> np.ndarray:
    """Sorted ``int64`` ref_ids with ``eligible[K] == True`` (default ``K=50``).

    These are the ref_ids every primary K is evaluated on (protocol section
    29): because ``eligible[K]`` is monotone in ``K``, the ``K=50`` set is
    exactly the intersection of the per-K eligible sets, so K=5/10/20/50 all
    run on identical ref_ids.
    """
    source = entries.entries if isinstance(entries, ManifestFile) else entries
    top = int(K)
    ref_ids = [int(entry.ref_id) for entry in source if bool(entry.eligible.get(top, False))]
    return np.asarray(sorted(ref_ids), dtype=np.int64)


def manifest_path(root: Union[str, Path], regime: str, split: str) -> Path:
    """``root/manifests/{regime}_{split}.jsonl`` (the frozen output layout)."""
    return Path(root) / "manifests" / f"{regime}_{split}.jsonl"


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def build_manifests(
    bank_path: Union[str, Path],
    refs: Union[str, Path, Iterable[Mapping[str, Any]]],
    instances: Union[str, Path, Mapping[str, Mapping[str, Any]]],
    regime: str,
    seed: int = MANIFEST_SEED,
    top_n: int = 64,
    *,
    coco: Optional[Union[str, Path, Mapping[str, Any]]] = None,
) -> ManifestFile:
    """Build the manifest of one regime over every expression in ``refs``.

    Parameters
    ----------
    bank_path:
        ``cache/proposals.h5``-style proposal bank (frozen group layout
        ``image_{image_id}`` / ``boxes`` / ``objectness``).  Reading is
        per image, so the fixed query-independent bank is loaded lazily.
    refs:
        Path to the UNC ``refs(unc).p`` pickle (loaded with
        :func:`ccg.data.refcoco.load_refs_pickle`) or the validated region
        records themselves.
    instances:
        Path to the RefCOCO+ ``instances.json`` (or the
        ``ann_id(str) -> {"image_id", "bbox", "category_id"}`` table) that
        joins every region's ``ann_id`` to its target box (``xywh``).
    regime:
        ``"random"`` or ``"same_category"`` (see the module docstring).
    seed:
        Frozen manifest seed; the per-ref rng is
        ``numpy.random.default_rng([seed, ref_id])``.
    top_n:
        Bank truncation (the frozen bank is ``N = 64`` per image).
    coco:
        COCO GT objects (``instances_train2014.json`` payload, path, or a
        decoded COCO dict) used for the same-category metadata.  Required for
        ``regime="same_category"``; without it the random regime records
        ``n_same_category_available = 0`` (metadata unavailable, not "none
        exist").

    Returns
    -------
    ManifestFile
        Entries in ``refs`` file order (deterministic, independent of the
        bank traversal order); ``meta`` carries the frozen provenance block.
    """
    regime = str(regime)
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
    seed = int(seed)
    top_n = int(top_n)
    if top_n < 2:
        raise ValueError(f"top_n must be >= 2 (K=5 needs at least 5 proposals), got {top_n}")
    if coco is None and regime == "same_category":
        raise ValueError(
            "regime='same_category' needs the COCO GT objects for the proposal "
            "categories; pass coco=<instances_train2014.json path or payload>"
        )

    if not isinstance(bank_path, (str, Path)):
        raise TypeError(f"bank_path must be a path to the proposal bank, got {type(bank_path)}")
    bank_file = Path(bank_path)
    if not bank_file.exists():
        raise FileNotFoundError(
            f"proposal bank {bank_file} not built yet - the frozen RPN bank is generated "
            "by the main agent first (cache/proposals.h5); re-run once it lands"
        )
    fingerprint = _bank_fingerprint(bank_file)

    records = _load_refs(refs)
    join_table = _load_join_table(instances)
    needed_images = sorted({_record_image_id(record) for record in records})
    gt_index = _load_gt_index(coco, needed_images)
    val_lookup, val_counts = _val_subsplits(records)

    grouped: Dict[int, List[Tuple[int, Mapping[str, Any]]]] = {}
    for position, record in enumerate(records):
        grouped.setdefault(_record_image_id(record), []).append((position, record))

    h5py = _import_h5py()
    built: List[Tuple[int, ManifestEntry]] = []
    missing_images: List[int] = []
    with h5py.File(bank_file, "r") as handle:
        for image_id, rows in grouped.items():
            node = handle.get(f"{_BANK_GROUP_PREFIX}{image_id}")
            if node is None:
                missing_images.append(int(image_id))
                continue
            bank = _bank_from_node(node, image_id, top_n)
            gt_objects = None if gt_index is None else gt_index.objects(image_id)
            for position, record in rows:
                entry = _build_entry(
                    record,
                    bank=bank,
                    join_table=join_table,
                    gt_objects=gt_objects,
                    regime=regime,
                    seed=seed,
                    top_n=top_n,
                    val_lookup=val_lookup,
                )
                built.append((position, entry))
    built.sort(key=lambda pair: pair[0])  # refs file order, not bank traversal order
    entries = [entry for _, entry in built]

    split_counts: Dict[str, int] = {}
    for entry in entries:
        split_counts[entry.split] = split_counts.get(entry.split, 0) + 1
    meta: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "regime": regime,
        "seed": seed,
        "top_n": top_n,
        "n_refs": len(entries),
        "n_images": len({int(entry.image_id) for entry in entries}),
        "bank_fingerprint": fingerprint,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_counts": {key: split_counts[key] for key in sorted(split_counts)},
        "iou_thresh": float(DEFAULT_IOU_THRESH),
        "n_missing_bank": len(missing_images),
        "missing_bank_images": missing_images[:32],
        "val_split_seed": int(DEFAULT_SPLIT_SEED),
        "val_split_counts": dict(val_counts),
        "coco_gt_objects": _coco_description(coco),
        "sources": {
            "refs": str(refs) if isinstance(refs, (str, Path)) else "<records>",
            "instances": (
                str(instances) if isinstance(instances, (str, Path)) else "<mapping>"
            ),
        },
    }
    return ManifestFile(regime=regime, meta=meta, entries=entries)


# ---------------------------------------------------------------------------
# construction internals
# ---------------------------------------------------------------------------
def _build_entry(
    record: Mapping[str, Any],
    *,
    bank: ProposalBank,
    join_table: Mapping[str, Mapping[str, Any]],
    gt_objects: Any,
    regime: str,
    seed: int,
    top_n: int,
    val_lookup: Mapping[int, str],
) -> ManifestEntry:
    """One frozen entry: assignment, ordering, eligibility, metadata."""
    for field in ("ref_id", "image_id", "split", "ann_id"):
        if record.get(field) is None:
            raise ValueError(f"ref record misses the required field {field!r}: keys={sorted(record)}")
    ref_id = int(record["ref_id"])
    image_id = int(record["image_id"])
    split = normalize_split_name(record["split"])
    if split not in FILE_SPLITS:
        raise ValueError(
            f"ref {ref_id}: split {record['split']!r} normalises to {split!r}, "
            f"outside {FILE_SPLITS}"
        )
    ann_id = int(record["ann_id"])
    info = join_table.get(str(ann_id))
    if info is None:
        raise ValueError(
            f"ref {ref_id}: ann_id {ann_id} is not in the instances join table"
        )
    if int(info["image_id"]) != image_id:
        raise ValueError(
            f"ref {ref_id}: ann_id {ann_id} belongs to image {int(info['image_id'])}, "
            f"not {image_id}"
        )
    x, y, width, height = (float(value) for value in info["bbox"])
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"ref {ref_id}: non-positive target box {info['bbox']!r}")
    gt_box = xywh_to_xyxy(np.asarray([x, y, width, height], dtype=np.float32))

    assignment = assign_target(bank, gt_box, iou_thresh=DEFAULT_IOU_THRESH)
    target_index = assignment.target_proposal_idx
    blocked = [int(value) for value in assignment.to_remove.tolist()]
    if target_index is not None:
        blocked.append(int(target_index))
    valid = _valid_pool(int(bank.N), blocked)
    n_valid = int(valid.size)

    categories: Optional[np.ndarray] = None
    if gt_objects is not None:
        categories, _ = assign_gt_category(
            bank.boxes, gt_objects.boxes, gt_objects.categories, iou_thresh=DEFAULT_IOU_THRESH
        )
    if categories is None or target_index is None:
        n_same_available = 0
        target_category = _UNKNOWN_CATEGORY
    else:
        target_category = int(categories[int(target_index)])
        # frozen audit helper: unknown (-1) never matches, target / equivalents excluded
        n_same_available = int(
            same_category_counts(categories, int(target_index), assignment.to_remove)
        )

    rng = np.random.default_rng([int(seed), ref_id])
    if regime == "random":
        ordering = valid[rng.permutation(n_valid)]
        n_same_used = {int(k): 0 for k in PRIMARY_KS}
        hard_fractions = {int(k): 0.0 for k in PRIMARY_KS}
    else:
        if n_same_available > 0 and categories is not None:
            same_mask = categories[valid] == target_category
            same = valid[same_mask]
            rest = valid[~same_mask]
            if int(same.size) != n_same_available:
                raise RuntimeError(
                    f"ref {ref_id}: same_category_counts ({n_same_available}) disagrees with "
                    f"the ordering split ({int(same.size)})"
                )
        else:
            same = np.empty(0, dtype=np.int64)
            rest = valid
        ordering = np.concatenate([same, rest[rng.permutation(int(rest.size))]])
        n_same_used = {
            int(k): int(min(n_same_available, int(k) - 1)) for k in PRIMARY_KS
        }
        hard_fractions = {
            int(k): float(min(n_same_available, int(k) - 1)) / float(int(k) - 1)
            for k in PRIMARY_KS
        }

    cap = int(top_n) - 1
    if ordering.size > cap:
        ordering = ordering[:cap]

    eligible = {
        int(k): bool(target_index is not None and n_valid >= int(k) - 1) for k in PRIMARY_KS
    }
    eval_split = None
    if split == SPLIT_VAL:
        eval_split = val_lookup.get(image_id)
        if eval_split is None:
            raise ValueError(f"ref {ref_id}: val image {image_id} is not covered by the cut")

    return ManifestEntry(
        ref_id=ref_id,
        image_id=image_id,
        split=split,
        target_index=target_index,
        distractor_order=np.ascontiguousarray(ordering, dtype=np.int32),
        n_valid_distractors=n_valid,
        n_target_equiv_removed=int(assignment.to_remove.size),
        n_same_category_available=n_same_available,
        eligible=eligible,
        n_same_used_by_K=n_same_used,
        hard_fraction_by_K=hard_fractions,
        target_max_iou=float(assignment.best_iou),
        eval_split=eval_split,
    )


def _valid_pool(n: int, blocked: Sequence[int]) -> np.ndarray:
    """Sorted bank indices in ``[0, n)`` minus ``blocked`` (target + equivalents)."""
    if not blocked:
        return np.arange(n, dtype=np.int64)
    blocked_arr = np.unique(np.asarray(list(blocked), dtype=np.int64))
    return np.setdiff1d(np.arange(n, dtype=np.int64), blocked_arr)


def _bank_from_node(node: Any, image_id: int, top_n: int) -> ProposalBank:
    """Truncated :class:`ProposalBank` from one h5 group (``boxes`` / ``objectness``)."""
    boxes = np.asarray(node["boxes"], dtype=np.float32).reshape(-1, 4)
    objectness = np.asarray(node["objectness"], dtype=np.float32).reshape(-1)
    if boxes.shape[0] == 0:
        raise ValueError(f"bank group {node.name} holds no proposals")
    if objectness.shape[0] != boxes.shape[0]:
        raise ValueError(
            f"bank group {node.name}: boxes {boxes.shape[0]} vs objectness "
            f"{objectness.shape[0]} row mismatch"
        )
    if boxes.shape[0] > top_n:
        boxes = boxes[:top_n]
        objectness = objectness[:top_n]
    return ProposalBank(image_id=int(image_id), boxes=boxes, objectness=objectness)


def _val_subsplits(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[int, str], Dict[str, int]]:
    """One image-level ``val_select`` / ``val_calib`` cut for all val images.

    The cut is computed once over the sorted image ids of every ``val``
    expression in ``records`` - with ``refs_by_image`` semantics, all
    expressions of an image land in the same sub-split.
    """
    val_images = sorted(
        {_record_image_id(record) for record in records
         if normalize_split_name(record.get("split")) == SPLIT_VAL}
    )
    if not val_images:
        return {}, {}
    select_ids, calib_ids = split_val_by_image(val_images, seed=DEFAULT_SPLIT_SEED)
    lookup: Dict[int, str] = {int(value): SPLIT_VAL_SELECT for value in select_ids.tolist()}
    lookup.update({int(value): SPLIT_VAL_CALIB for value in calib_ids.tolist()})
    counts = {"val_select": int(select_ids.size), "val_calib": int(calib_ids.size)}
    return lookup, counts


# ---------------------------------------------------------------------------
# loading internals
# ---------------------------------------------------------------------------
def _load_refs(
    refs: Union[str, Path, Iterable[Mapping[str, Any]]],
) -> List[Mapping[str, Any]]:
    if isinstance(refs, (str, Path)):
        return list(load_refs_pickle(Path(refs)))
    records = list(refs)
    if not records:
        raise ValueError("refs is empty - nothing to build")
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"refs[{position}] must be a region mapping, got {type(record)}")
    return records


def _load_join_table(
    instances: Union[str, Path, Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Mapping[str, Any]]:
    if isinstance(instances, (str, Path)):
        return dict(load_instances_json(Path(instances)))
    if isinstance(instances, Mapping):
        table: Dict[str, Mapping[str, Any]] = {}
        for key, value in instances.items():
            if not isinstance(value, Mapping):
                raise TypeError(f"instances[{key!r}] must be a mapping, got {type(value)}")
            table[str(key)] = value
        if not table:
            raise ValueError("instances join table is empty")
        return table
    raise TypeError(f"instances must be a path or mapping, got {type(instances)}")


def _load_gt_index(
    coco: Optional[Union[str, Path, Mapping[str, Any]]], needed_images: Sequence[int]
) -> Optional[CocoIndex]:
    if coco is None:
        return None
    source = ""
    if isinstance(coco, (str, Path)):
        path = Path(coco)
        if not path.exists():
            raise FileNotFoundError(
                f"COCO GT annotations {path} not found - needed for the same-category "
                "metadata (proposal -> highest-IoU GT object)"
            )
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        source = str(path)
    elif isinstance(coco, Mapping):
        payload = coco
        source = "<payload>"
    else:
        raise TypeError(f"coco must be a path or a COCO payload mapping, got {type(coco)}")
    return build_coco_index(payload, source=source, only_image_ids=list(needed_images))


def _record_image_id(record: Mapping[str, Any]) -> int:
    if record.get("image_id") is None:
        raise ValueError(f"ref record misses image_id: keys={sorted(record)}")
    return int(record["image_id"])


def _coco_description(coco: Optional[Union[str, Path, Mapping[str, Any]]]) -> Optional[str]:
    if coco is None:
        return None
    if isinstance(coco, (str, Path)):
        return str(coco)
    return "<payload>"


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "h5py is required to read the proposal bank (cache/proposals.h5)"
        ) from exc
    return h5py


def _bank_fingerprint(bank_file: Path) -> str:
    """Lightweight sha256 fingerprint: file size + up to 8 MiB of prefix.

    The bank can be hundreds of MB; hashing the full file on every build is
    wasteful, while size + prefix already changes on any rebuild that touches
    the stored groups.  Recorded in ``meta["bank_fingerprint"]``.
    """
    digest = hashlib.sha256()
    digest.update(str(int(bank_file.stat().st_size)).encode("ascii"))
    with bank_file.open("rb") as handle:
        digest.update(handle.read(_FINGERPRINT_PREFIX_BYTES))
    return digest.hexdigest()


def _json_fallback(value: Any) -> Any:
    """JSON encoder fallback for numpy scalars/arrays (meta provenance)."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"meta value {value!r} is not JSON serialisable")
