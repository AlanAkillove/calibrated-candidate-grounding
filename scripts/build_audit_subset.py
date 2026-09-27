"""Deterministic image-level audit subset builder - reproduces the frozen csv.

This is the reproducible implementation of the frozen sampling algorithm whose
reference output is ``data/audit_subset.csv`` (1,500 images: 1,000 RefCOCO+
``train`` + 500 ``val_select``, all from COCO ``train2014``).  It ports the
one-off canonical generator ``tools/build_audit_subset.py`` with identical
semantics; with the default parameters it reproduces the frozen csv exactly
(same image ids, same labels, same row order, same columns - verified
byte-for-byte on 2026-09-27, and re-checked against the csv by
``tests/test_audit_subset.py`` whenever the raw data is present).

Frozen algorithm (defaults: seed=20260927, n_train=1000, n_val=500)
-------------------------------------------------------------------
1. ``train_pool`` / ``val_pool`` = sorted unique image ids of split ``train`` /
   ``val`` in ``refs(unc).p`` (``testA`` / ``testB`` images are never eligible).
2. ``val_select`` = ``val_pool[i]`` for the first half of
   ``numpy.random.default_rng(seed + 1).permutation(len(val_pool))``, re-sorted
   ascending (the same image-level half the training protocol cuts through
   :mod:`ccg.data.splits`).
3. **one** generator ``rng = numpy.random.default_rng(seed)`` draws ``n_train``
   ids from ``train_pool`` and *then* ``n_val`` ids from ``val_select`` via
   ``rng.choice(len(pool), size=n, replace=False)`` - the order of the two draws
   and the call shape are part of the freeze (any "equivalent" refactor changes
   the sample).
4. rows carry ``image_id, file_name, coco_split, refcoco_split, ref_count``:
   ``file_name`` / ``coco_split`` are joined from the ``images[]`` of
   ``instances_val2014.json`` then ``instances_train2014.json`` under
   ``<raw>/annotations`` (val first, exactly as in the freeze), and ``ref_count``
   counts the RefCOCO+ region records of that image.
5. rows are written sorted by ``(coco_split, image_id)``.

Pools that are too small raise (``ValueError``) instead of being padded - only
the *error path* is friendlier than the one-off script, the sample itself is
bit-identical for the shipped data.

Usage
-----
    python scripts/build_audit_subset.py             # refuses to overwrite (exit 0)
    python scripts/build_audit_subset.py --force     # regenerate the frozen csv
    python scripts/build_audit_subset.py --seed 1 --n-train 10 --n-val 5 \
        --out /tmp/smoke.csv --raw data/raw --force

Changing the defaults is a protocol amendment, not an experiment knob.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ccg.data.refcoco import load_refs_pickle  # noqa: E402  (pure, no torch)

__all__ = [
    "DEFAULT_SEED",
    "DEFAULT_N_TRAIN",
    "DEFAULT_N_VAL",
    "DEFAULT_RAW_DIR",
    "DEFAULT_OUT_CSV",
    "CSV_FIELDS",
    "COCO_INSTANCE_FILES",
    "REFEROCO_PICKLE_CANDIDATES",
    "find_refs_pickle",
    "unique_split_image_ids",
    "select_val_subset",
    "count_refs_per_image",
    "draw_audit_images",
    "load_coco_image_manifest",
    "build_subset_rows",
    "write_subset_csv",
    "build_parser",
    "main",
]

#: Seed of the frozen draw (a protocol amendment date, not an experiment knob).
DEFAULT_SEED = 20260927
DEFAULT_N_TRAIN = 1000
DEFAULT_N_VAL = 500
DEFAULT_RAW_DIR = _REPO_ROOT / "data" / "raw"
DEFAULT_OUT_CSV = _REPO_ROOT / "data" / "audit_subset.csv"

#: Exact column set of the frozen csv, in writing order.
CSV_FIELDS: Tuple[str, ...] = ("image_id", "file_name", "coco_split", "refcoco_split", "ref_count")

#: ``(coco_split label, instances file)`` in the frozen manifest order (val first).
COCO_INSTANCE_FILES: Tuple[Tuple[str, str], ...] = (
    ("val2014", "instances_val2014.json"),
    ("train2014", "instances_train2014.json"),
)

#: ``refs(unc).p`` locations under ``--raw``, in probe order.
REFEROCO_PICKLE_CANDIDATES: Tuple[str, ...] = (
    "refcoco+/refcoco+/refs(unc).p",
    "refcoco+/refs(unc).p",
)


# ---------------------------------------------------------------------------
# frozen selection algorithm
# ---------------------------------------------------------------------------
def find_refs_pickle(raw: str | Path) -> Optional[Path]:
    """First existing ``refs(unc).p`` under ``raw`` (``None`` when absent)."""
    root = Path(raw)
    for relative in REFEROCO_PICKLE_CANDIDATES:
        candidate = root / relative
        if candidate.exists():
            return candidate
    return None


def unique_split_image_ids(refs: Sequence[Mapping], split: str) -> List[int]:
    """Sorted unique ``int(image_id)`` of the records labelled ``split``.

    Precondition: ``refs`` went through :func:`ccg.data.refcoco.load_refs_pickle`
    (the records carry ``image_id`` / ``split``).
    """
    return sorted({int(record["image_id"]) for record in refs if record["split"] == split})


def select_val_subset(val_pool: Sequence[int], seed: int) -> List[int]:
    """Frozen ``val_select``: first half of the ``seed + 1`` permutation, sorted.

    A pool with fewer than two images yields an empty ``val_select`` (the
    frozen generator's ``len // 2`` semantics; nothing is invented).
    """
    rng = np.random.default_rng(int(seed) + 1)
    perm = rng.permutation(len(val_pool))
    return sorted(val_pool[int(i)] for i in perm[: len(val_pool) // 2])


def count_refs_per_image(refs: Sequence[Mapping]) -> Dict[int, int]:
    """Region records per image id (the ``ref_count`` column)."""
    counts: Dict[int, int] = {}
    for record in refs:
        image_id = int(record["image_id"])
        counts[image_id] = counts.get(image_id, 0) + 1
    return counts


def draw_audit_images(
    refs: Sequence[Mapping],
    *,
    seed: int = DEFAULT_SEED,
    n_train: int = DEFAULT_N_TRAIN,
    n_val: int = DEFAULT_N_VAL,
) -> Tuple[List[Tuple[int, str]], Dict[int, int]]:
    """The frozen two-draw selection: ``((image_id, refcoco_split) pairs, ref counts)``.

    The pairs come in *draw order* (the train block first, then the val block) -
    the csv is sorted later - and the counts cover every image of ``refs``.
    """
    train_pool = unique_split_image_ids(refs, "train")
    val_pool = unique_split_image_ids(refs, "val")
    val_select = select_val_subset(val_pool, seed)
    if int(n_train) <= 0 or int(n_val) <= 0:
        raise ValueError(f"n_train and n_val must be positive, got {n_train} and {n_val}")
    if int(n_train) > len(train_pool):
        raise ValueError(
            f"n_train={n_train} exceeds the train pool ({len(train_pool)} images); "
            "the frozen algorithm never pads"
        )
    if int(n_val) > len(val_select):
        raise ValueError(
            f"n_val={n_val} exceeds val_select ({len(val_select)} of {len(val_pool)} val images)"
        )
    rng = np.random.default_rng(int(seed))
    train_chosen = [
        train_pool[int(i)] for i in rng.choice(len(train_pool), size=int(n_train), replace=False)
    ]
    val_chosen = [
        val_select[int(i)] for i in rng.choice(len(val_select), size=int(n_val), replace=False)
    ]
    pairs: List[Tuple[int, str]] = [(image_id, "train") for image_id in train_chosen]
    pairs += [(image_id, "val_select") for image_id in val_chosen]
    return pairs, count_refs_per_image(refs)


# ---------------------------------------------------------------------------
# manifest join + outputs
# ---------------------------------------------------------------------------
def load_coco_image_manifest(annotations_dir: str | Path) -> Dict[int, Tuple[str, str]]:
    """``image_id -> (file_name, coco_split)`` from the COCO ``instances_*.json`` files.

    Reads ``instances_val2014.json`` first and ``instances_train2014.json``
    second (frozen overwrite order - COCO splits are disjoint, so it only fixes
    the behaviour for a malformed manifest).  Missing files raise: deriving the
    file name instead would silently replace a lookup by a convention.
    """
    directory = Path(annotations_dir)
    manifest: Dict[int, Tuple[str, str]] = {}
    for split_label, file_name in COCO_INSTANCE_FILES:
        path = directory / file_name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - the COCO instances files provide the file_name / "
                "coco_split columns"
            )
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        images = payload.get("images") if isinstance(payload, Mapping) else None
        if not isinstance(images, list) or not images:
            raise ValueError(f"{path}: no images[] list")
        for image in images:
            manifest[int(image["id"])] = (str(image["file_name"]), split_label)
    return manifest


def build_subset_rows(
    *,
    refs_path: str | Path,
    annotations_dir: str | Path,
    seed: int = DEFAULT_SEED,
    n_train: int = DEFAULT_N_TRAIN,
    n_val: int = DEFAULT_N_VAL,
) -> List[dict]:
    """The frozen csv rows (sorted by ``(coco_split, image_id)``) - pure function of the files."""
    refs = load_refs_pickle(refs_path)
    pairs, ref_counts = draw_audit_images(refs, seed=seed, n_train=n_train, n_val=n_val)
    manifest = load_coco_image_manifest(annotations_dir)
    missing = sorted({image_id for image_id, _ in pairs if image_id not in manifest})
    if missing:
        raise RuntimeError(
            f"{len(missing)} sampled image ids are absent from the COCO instances manifest, "
            f"e.g. {missing[:5]} - the frozen subset cannot be built from this data"
        )
    rows = [
        {
            "image_id": int(image_id),
            "file_name": manifest[image_id][0],
            "coco_split": manifest[image_id][1],
            "refcoco_split": label,
            "ref_count": int(ref_counts.get(image_id, 0)),
        }
        for image_id, label in pairs
    ]
    rows.sort(key=lambda row: (row["coco_split"], row["image_id"]))
    return rows


def write_subset_csv(rows: Sequence[Mapping], path: str | Path) -> Path:
    """Write the rows with the frozen column set / dialect (``csv.DictWriter``)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    return out


def _count_data_rows(path: Path) -> Optional[int]:
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"raw data root holding refcoco+/ and annotations/ (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="frozen audit seed")
    parser.add_argument(
        "--n-train",
        type=int,
        default=DEFAULT_N_TRAIN,
        dest="n_train",
        help="train images in the subset (frozen default 1000)",
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=DEFAULT_N_VAL,
        dest="n_val",
        help="val_select images in the subset (frozen default 500)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help=f"output csv (default: {DEFAULT_OUT_CSV})",
    )
    parser.add_argument("--force", action="store_true", help="overwrite --out when it exists")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    out = Path(args.out)
    if out.exists() and not args.force:
        n_rows = _count_data_rows(out)
        suffix = "" if n_rows is None else f" ({n_rows} data rows)"
        print(
            f"[build_audit_subset] {out} already exists{suffix}; refusing to overwrite. "
            "Pass --force to regenerate it."
        )
        return 0

    raw = Path(args.raw)
    refs_path = find_refs_pickle(raw)
    if refs_path is None:
        print(
            f"[build_audit_subset] no refs(unc).p under {raw} (looked in "
            f"{', '.join(REFEROCO_PICKLE_CANDIDATES)}); download/unzip RefCOCO+ first "
            "(tools/download_data.py) or pass --raw explicitly."
        )
        return 1

    try:
        rows = build_subset_rows(
            refs_path=refs_path,
            annotations_dir=raw / "annotations",
            seed=int(args.seed),
            n_train=int(args.n_train),
            n_val=int(args.n_val),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[build_audit_subset] error: {exc}")
        return 1

    write_subset_csv(rows, out)
    n_train_rows = sum(1 for row in rows if row["refcoco_split"] == "train")
    n_val_rows = sum(1 for row in rows if row["refcoco_split"] == "val_select")
    print(
        f"[build_audit_subset] wrote {out}: {len(rows)} images "
        f"(train={n_train_rows}, val_select={n_val_rows}; seed={args.seed}, "
        f"n_train={args.n_train}, n_val={args.n_val})."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
