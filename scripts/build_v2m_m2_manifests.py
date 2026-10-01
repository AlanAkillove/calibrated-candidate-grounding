#!/usr/bin/env python
"""V2-M M2 manifest freeze: derive and freeze the competition-curriculum level cells.

M2 reuses the frozen ``manifests-v1`` (``cache/manifests``, seed 20260927)
exactly - nothing is resampled or reordered:

* the training / tuning levels ``m in {0, 2, 4}`` are the frozen
  ``same_category`` level construction

      ``C_10 = [target] + same_order[:m] + first (10 - 1 - m) of the rest shuffle``

  (``ccg.semantic.hard.HardCohort.level_indices`` / ``level_sample``, K = 10 -
  the construction default and the only K ever materialised for levels; the
  frozen Phase 1F ``expb`` cells are K = 10);
* the unseen test level ``m = 8`` is the frozen Phase 1F ``expb_m8`` cell
  (K = 10, ``same8`` cohort, ``testA + testB`` rows) - never redefined, because
  ``K = 5`` cannot host 8 same-category distractors.

This script

1. derives the ``reliability_train`` / ``reliability_tune`` row tables of every
   training level from the frozen value manifest (pure function of the frozen
   inputs; eligibility = target present, ``eligible[10]``,
   ``n_same_category_available >= m`` and enough rest supply);
2. verifies the frozen test-side construction **bit-for-bit** against
   ``expb_m{0,2,4,8}_candidates.npz``;
3. proves the ``train / tune / test`` image disjointness required by the M2
   protocol;
4. writes ``m2_curriculum/manifests/val_level_m{0,2,4}.npz`` (frozen row
   tables) plus ``manifest_freeze.json`` carrying every sha256 - the freeze
   record the runner must verify before training.

Nothing here scores or trains: the output is data provenance only.

Usage
-----
    python scripts/build_v2m_m2_manifests.py            # build + freeze
    python scripts/build_v2m_m2_manifests.py --verify   # re-derive and compare
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from ccg.data.manifests import ManifestFile  # noqa: E402
from ccg.semantic import data as sdata  # noqa: E402
from ccg.semantic import hard as shard  # noqa: E402

__all__ = ["build_parser", "main", "derive_level_tables", "load_val_cohort"]

OUT_ROOT = Path("results/v2_local_competition")
M2_DIR = OUT_ROOT / "m2_curriculum"
MAN_DIR = M2_DIR / "manifests"
FREEZE_PATH = MAN_DIR / "manifest_freeze.json"

PHASE05_DIR = sdata.PHASE05_FEATURES_DIR
MANIFESTS = Path("cache/manifests")
SPLIT_MANIFEST = Path("results/phase05_score_sufficiency/split_manifest.json")
EXPB_MANIFESTS = Path("results/phase1f_hard_semantic/manifests")

#: Training / tuning curriculum levels (preregistered; never searched).
LEVELS_TRAIN: Tuple[int, ...] = (0, 2, 4)
#: Frozen evaluation levels of the severity curve (m = 8 is the unseen primary).
LEVELS_TEST: Tuple[int, ...] = (0, 2, 4, 8)
#: Candidate-set size of every M2 level (frozen level construction, Phase 1F expb cells).
K_LEVEL = 10

VAL_MANIFEST_SOURCES = (
    "same_category_val.jsonl",
    "same_category_val.meta.json",
    "random_val.jsonl",
    "random_val.meta.json",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _git(args: Sequence[str]) -> Optional[str]:
    try:
        proc = subprocess.run(["git", *args], cwd=str(_REPO), capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


# ---------------------------------------------------------------------------
# cohort + derivation
# ---------------------------------------------------------------------------
def load_val_cohort() -> shard.HardCohort:
    """Canonical-cohort rows joined to the frozen ``val`` manifests (val_calib)."""
    cohort = sdata.load_phase05_cohort(5, features_dir=PHASE05_DIR)
    entries: Dict[str, Dict[int, Any]] = {}
    for regime in ("random", "same_category"):
        table: Dict[int, Any] = {}
        manifest = ManifestFile.load(MANIFESTS / f"{regime}_val.jsonl")
        for entry in manifest.entries:
            table[int(entry.ref_id)] = entry
        entries[regime] = table

    ref_id = np.asarray(cohort["ref_id"], dtype=np.int64)
    eval_split = np.asarray(cohort["eval_split"])
    nsame = np.full(ref_id.size, -1, dtype=np.int64)
    missing: List[int] = []
    for pos in np.flatnonzero(np.isin(eval_split, ("val_select", "val_calib"))).tolist():
        entry = entries["same_category"].get(int(ref_id[pos]))
        if entry is None:
            missing.append(int(ref_id[pos]))
        else:
            nsame[pos] = int(entry.n_same_category_available)
    if missing:
        raise AssertionError(
            f"{len(missing)} val refs missing from same_category_val.jsonl (first: {missing[0]})"
        )
    masks = {
        "base": nsame >= 0,
        "same2": nsame >= 2,
        "same4": nsame >= 4,
        "same8": nsame >= 8,
    }
    return shard.HardCohort(
        sentence_id=np.asarray(cohort["sentence_id"], dtype=np.int64),
        ref_id=ref_id,
        image_id=np.asarray(cohort["image_id"], dtype=np.int64),
        eval_split=eval_split,
        nsame=nsame,
        masks=masks,
        entries=entries,
    )


def _level_feasible(entry: Any, m: int, k: int = K_LEVEL) -> bool:
    """Frozen level-``m`` feasibility of one ref (see module docstring)."""
    if entry.target_index is None or not bool(entry.eligible.get(int(k), False)):
        return False
    order = np.asarray(entry.distractor_order)
    n_avail = int(entry.n_same_category_available)
    return n_avail >= int(m) and int(order.size) >= n_avail + (int(k) - 1 - int(m))


def derive_level_tables(
    *,
    k: int = K_LEVEL,
    log: Optional[Any] = None,
) -> Dict[str, Any]:
    """Deterministic derivation of the frozen val level cells (train + tune).

    Returns ``{"cohort": HardCohort, "split": ReliabilitySplit, "tables": {...}}``
    with ``tables[f"m{m}_{side}"] = {"rows", "indices"}``.
    """
    emit = log if log is not None else (lambda _message: None)
    cohort = load_val_cohort()
    split = sdata.load_split_from_manifest(SPLIT_MANIFEST)
    sdata.assert_split_matches_fresh(split, cohort.eval_split, cohort.image_id)
    train_mask = split.row_mask(cohort.eval_split, cohort.image_id, kind="reliability_train")
    tune_mask = split.row_mask(cohort.eval_split, cohort.image_id, kind="reliability_tune")
    val = np.asarray(cohort.eval_split) == "val_calib"
    if not (bool(np.all(val[train_mask])) and bool(np.all(val[tune_mask]))):
        raise AssertionError("reliability train/tune rows leave val_calib -- split drift")

    tables: Dict[str, Any] = {}
    for m in LEVELS_TRAIN:
        feasible = np.zeros(len(cohort), dtype=bool)
        for row in np.flatnonzero(val).tolist():
            entry = cohort.entries["same_category"].get(int(cohort.ref_id[row]))
            feasible[row] = entry is not None and _level_feasible(entry, m, k)
        for side, mask in (("train", train_mask), ("tune", tune_mask)):
            rows = np.flatnonzero(feasible & mask)
            indices = (
                np.stack([cohort.level_indices(int(r), m, k) for r in rows.tolist()])
                if rows.size
                else np.empty((0, k), dtype=np.int64)
            )
            tables[f"m{m}_{side}"] = {"rows": rows, "indices": np.asarray(indices, dtype=np.int64)}
            emit(f"[m2-build] level m={m} {side}: {rows.size} rows / {np.unique(cohort.image_id[rows]).size} images")
    return {"cohort": cohort, "split": split, "tables": tables}


# ---------------------------------------------------------------------------
# frozen test-side equivalence
# ---------------------------------------------------------------------------
def verify_test_side(cohort_hc: shard.HardCohort, *, k: int = K_LEVEL) -> Dict[str, Any]:
    """Prove the level construction reproduces the frozen ``expb`` cells exactly."""
    rows = np.flatnonzero(np.asarray(cohort_hc.masks["same8"], dtype=bool))
    ids = np.asarray(cohort_hc.sentence_id[rows], dtype=np.int64)
    results: Dict[str, Any] = {}
    for m in LEVELS_TEST:
        path = EXPB_MANIFESTS / f"expb_m{int(m)}_candidates.npz"
        with np.load(path) as frozen:
            frozen_ids = np.asarray(frozen["sentence_id"], dtype=np.int64)
            frozen_ci = np.asarray(frozen["candidate_indices"], dtype=np.int64)
        if not np.array_equal(ids, frozen_ids):
            raise AssertionError(f"expb_m{m}: sentence_id order differs from the same8 cohort rebuild")
        mine = np.stack([cohort_hc.level_indices(int(r), int(m), int(k)) for r in rows.tolist()])
        if not np.array_equal(mine, frozen_ci):
            bad = int(np.flatnonzero((mine != frozen_ci).any(axis=1))[0])
            raise AssertionError(f"expb_m{m}: candidate construction differs at row {bad}")
        results[f"expb_m{m}"] = {
            "rows": int(rows.size),
            "sha256": _sha256_file(path),
            "construction_match": True,
        }
    return results


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------
def build(*, log: Optional[Any] = None) -> int:
    started = time.perf_counter()
    emit = log if log is not None else (lambda message: print(message, flush=True))
    emit("[m2-build] deriving the frozen val level tables (manifests-v1, seed 20260927)")
    derived = derive_level_tables(log=emit)
    cohort = derived["cohort"]
    tables = derived["tables"]

    cohort_hc = shard.load_hard_cohort(features_dir=PHASE05_DIR, manifests_root=MANIFESTS, log=emit)
    test_side = verify_test_side(cohort_hc)
    emit("[m2-build] test-side construction reproduces expb_m{0,2,4,8} bit-for-bit")

    rows_same8 = np.flatnonzero(np.asarray(cohort_hc.masks["same8"], dtype=bool))
    test_images = set(np.asarray(cohort_hc.image_id[rows_same8]).tolist())
    test_rows = set(np.asarray(cohort_hc.sentence_id[rows_same8]).tolist())
    train_mask = derived["split"].row_mask(cohort.eval_split, cohort.image_id, kind="reliability_train")
    tune_mask = derived["split"].row_mask(cohort.eval_split, cohort.image_id, kind="reliability_tune")
    train_images = set(np.asarray(cohort.image_id[train_mask]).tolist())
    tune_images = set(np.asarray(cohort.image_id[tune_mask]).tolist())
    train_rows = set(np.asarray(cohort.sentence_id[train_mask]).tolist())
    tune_rows = set(np.asarray(cohort.sentence_id[tune_mask]).tolist())
    disjointness = {
        "train_tune_images": len(train_images & tune_images),
        "train_test_images": len(train_images & test_images),
        "tune_test_images": len(tune_images & test_images),
        "train_test_rows": len(train_rows & test_rows),
        "tune_test_rows": len(tune_rows & test_rows),
    }
    if any(disjointness.values()):
        raise AssertionError(f"M2 train/tune/test isolation violated: {disjointness}")
    emit(f"[m2-build] isolation ok: train={len(train_images)} tune={len(tune_images)} test={len(test_images)} images, pairwise disjoint")

    MAN_DIR.mkdir(parents=True, exist_ok=True)
    val_tables: Dict[str, Any] = {}
    for m in LEVELS_TRAIN:
        path = MAN_DIR / f"val_level_m{int(m)}.npz"
        payload: Dict[str, np.ndarray] = {}
        for side in ("train", "tune"):
            entry = tables[f"m{int(m)}_{side}"]
            rows = entry["rows"]
            payload.update(
                {
                    f"{side}__sentence_id": np.asarray(cohort.sentence_id[rows], dtype=np.int64),
                    f"{side}__ref_id": np.asarray(cohort.ref_id[rows], dtype=np.int64),
                    f"{side}__image_id": np.asarray(cohort.image_id[rows], dtype=np.int64),
                    f"{side}__candidate_indices": np.asarray(entry["indices"], dtype=np.int64),
                    f"{side}__target_index": np.asarray(entry["indices"][:, 0], dtype=np.int64),
                }
            )
        np.savez_compressed(path, **payload)
        val_tables[f"m{int(m)}"] = {
            "file": path.name,
            "sha256": _sha256_file(path),
            "train_rows": int(tables[f"m{int(m)}_train"]["rows"].size),
            "train_images": int(np.unique(cohort.image_id[tables[f"m{int(m)}_train"]["rows"]]).size),
            "tune_rows": int(tables[f"m{int(m)}_tune"]["rows"].size),
            "tune_images": int(np.unique(cohort.image_id[tables[f"m{int(m)}_tune"]["rows"]]).size),
        }
        emit(
            f"[m2-build] frozen {path.name}: train {val_tables[f'm{int(m)}']['train_rows']} rows, "
            f"tune {val_tables[f'm{int(m)}']['tune_rows']} rows (sha256 {val_tables[f'm{int(m)}']['sha256'][:12]}...)"
        )

    sources = {name: _sha256_file(MANIFESTS / name) for name in VAL_MANIFEST_SOURCES}
    sources["split_manifest.json"] = _sha256_file(SPLIT_MANIFEST)
    payload = {
        "artifact": "v2m_m2_manifest_freeze",
        "rule": (
            "M2 levels reuse the frozen manifests-v1 exactly; the val m=2/m=4 tables are a "
            "deterministic derivation of same_category_val.jsonl under the Phase1F level "
            "construction, the test cells are the frozen Phase1F expb manifests. No resampling."
        ),
        "k_level": int(K_LEVEL),
        "levels_train": list(LEVELS_TRAIN),
        "levels_test": list(LEVELS_TEST),
        "construction": (
            "[target] + same_order[:m] + first (k-1-m) of the manifest rest shuffle "
            "(HardCohort.level_indices / level_sample; K=10)"
        ),
        "eligibility": (
            "target present AND eligible[10] AND n_same_category_available >= m AND "
            "distractor_order.size >= n_avail + (10-1-m)"
        ),
        "k_mapping": {
            "train": {f"m{int(m)}": int(K_LEVEL) for m in LEVELS_TRAIN},
            "test": {f"m{int(m)}": int(K_LEVEL) for m in LEVELS_TEST},
            "note": (
                "the M2 preregistration fixes m=8 to the frozen K=10 same8 cell "
                "(K=5 can host at most 4 distractors and is never redefined)"
            ),
        },
        "sources": sources,
        "test_side_verification": test_side,
        "val_tables": val_tables,
        "disjointness": disjointness,
        "git_head": _git(["rev-parse", "HEAD"]),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_sec": None,  # filled below
    }
    payload["runtime_sec"] = round(time.perf_counter() - started, 1)
    _write_json(FREEZE_PATH, payload)
    emit(f"[m2-build] freeze record written: {FREEZE_PATH}")
    emit("[m2-build] K reconciliation (checked against the M2 preregistration + Phase 1F cells)")
    for m in LEVELS_TRAIN:
        emit(f"[m2-build]   train m{m} -> K={K_LEVEL}")
    emit(f"[m2-build]   test  m8 -> K={K_LEVEL} (frozen expb_m8 cell)")
    emit(f"[m2-build]   test  severity m0/m2/m4 -> K={K_LEVEL} (frozen expb cells, same rows)")
    emit(
        "V2M_M2_MANIFEST_FREEZE_OK "
        f"levels_train={list(LEVELS_TRAIN)} levels_test={list(LEVELS_TEST)} K={K_LEVEL} "
        f"runtime={payload['runtime_sec']}s"
    )
    return 0


def verify() -> int:
    """Re-derive the tables and compare against the stored freeze record."""
    if not FREEZE_PATH.exists():
        raise SystemExit(f"freeze record missing: {FREEZE_PATH}; run the build first")
    stored = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    derived = derive_level_tables()
    cohort = derived["cohort"]
    for m in LEVELS_TRAIN:
        path = MAN_DIR / f"val_level_m{int(m)}.npz"
        digest = _sha256_file(path)
        if digest != stored["val_tables"][f"m{int(m)}"]["sha256"]:
            raise AssertionError(f"val_level_m{m}.npz hash drift: {digest}")
        with np.load(path) as frozen:
            for side in ("train", "tune"):
                expected = derived["tables"][f"m{int(m)}_{side}"]
                rows = expected["rows"]
                if not np.array_equal(np.asarray(frozen[f"{side}__sentence_id"]), np.asarray(cohort.sentence_id[rows])):
                    raise AssertionError(f"m{m}/{side}: sentence_id drift")
                if not np.array_equal(np.asarray(frozen[f"{side}__candidate_indices"]), np.asarray(expected["indices"])):
                    raise AssertionError(f"m{m}/{side}: candidate_indices drift")
    print("V2M_M2_MANIFEST_VERIFY_OK (tables are deterministic and match the freeze record)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="re-derive and compare against the freeze record")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify:
        return verify()
    return build()


if __name__ == "__main__":
    raise SystemExit(main())
