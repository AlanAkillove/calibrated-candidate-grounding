"""Unit tests for :mod:`ccg.reliability.data` (synthetic npz + real-data smoke).

The synthetic tests write tiny B3-style / cosine-style ``npz`` files under
``tmp_path`` and exercise the loaders, the frozen image-level split and the
manifest writer.  A separate ``skipif`` test smoke-checks the *real* Phase 0
artifacts (20,799 common-cohort rows, cosine filtered to the same sentence set,
per-seed scorer independence).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ccg.reliability import data
from ccg.reliability.data import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_FRAC,
    SCORER_IDS,
    ReliabilitySplit,
    build_reliability_split,
    common_sentence_ids,
    load_scorer_scores,
    write_split_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_B3_ROOT = REPO_ROOT / "results" / "phase0b_independent"
REAL_COSINE_ROOT = REPO_ROOT / "results" / "phase0a_cosine"
REAL_B3_K5 = REAL_B3_ROOT / "seed_1" / "raw_scores" / "K5.npz"


# ---------------------------------------------------------------------------
# synthetic writers
# ---------------------------------------------------------------------------
def _write_b3(
    root: Path,
    seed_dir: str,
    k: int,
    *,
    scores: np.ndarray,
    sentence_id: np.ndarray,
    ref_id: np.ndarray,
    image_id: np.ndarray,
    eval_split: np.ndarray,
    target_local: np.ndarray,
) -> Path:
    out = root / seed_dir / "raw_scores"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"K{k}.npz"
    np.savez(
        path,
        scores=np.asarray(scores, dtype=np.float32),
        sentence_id=np.asarray(sentence_id, dtype=np.int64),
        ref_id=np.asarray(ref_id, dtype=np.int64),
        image_id=np.asarray(image_id, dtype=np.int64),
        eval_split=np.asarray(eval_split, dtype="<U10"),
        target_local=np.asarray(target_local, dtype=np.int32),
    )
    return path


def _write_cosine(
    root: Path,
    k: int,
    *,
    raw_scores: np.ndarray,
    sentence_ids: np.ndarray,
    ref_ids: np.ndarray,
    image_ids: np.ndarray,
    split_codes: np.ndarray,
    target_local: np.ndarray,
) -> Path:
    out = root / "raw_predictions"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"K{k}.npz"
    np.savez(
        path,
        raw_scores=np.asarray(raw_scores, dtype=np.float32),
        sentence_ids=np.asarray(sentence_ids, dtype=np.int64),
        ref_ids=np.asarray(ref_ids, dtype=np.int64),
        image_ids=np.asarray(image_ids, dtype=np.int64),
        split_codes=np.asarray(split_codes, dtype=np.int8),
        target_local=np.asarray(target_local, dtype=np.int32),
    )
    return path


def _sha(ids) -> str:
    joined = ",".join(str(int(x)) for x in sorted(int(i) for i in ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# (a) split determinism / disjointness / rounded 747 image split
# ---------------------------------------------------------------------------
def test_build_split_is_deterministic_and_rounded():
    n_images = 747
    image_id = np.arange(n_images, dtype=np.int64)
    eval_split = np.array(["val_calib"] * n_images, dtype="<U10")

    first = build_reliability_split(eval_split, image_id)
    second = build_reliability_split(eval_split, image_id)

    # int(round(0.70 * 747)) = 523 -> 523 / 224
    assert first.train_images.size == 523
    assert first.tune_images.size == 224
    assert first.seed == DEFAULT_SPLIT_SEED
    assert first.train_frac == DEFAULT_TRAIN_FRAC
    assert np.array_equal(first.train_images, second.train_images)
    assert np.array_equal(first.tune_images, second.tune_images)

    # image-disjoint + exhaustive over the val_calib image universe
    assert np.intersect1d(first.train_images, first.tune_images).size == 0
    union = np.union1d(first.train_images, first.tune_images)
    assert np.array_equal(union, np.unique(image_id))
    # sorted storage
    assert np.all(np.diff(first.train_images) > 0)
    assert np.all(np.diff(first.tune_images) > 0)


def test_build_split_only_uses_val_calib_images_and_changes_with_seed():
    eval_split = np.array(
        ["val_calib"] * 6 + ["testA"] * 3 + ["val_select", "testB"], dtype="<U10"
    )
    image_id = np.arange(11, dtype=np.int64)

    split = build_reliability_split(eval_split, image_id, seed=7, train_frac=0.5)
    assert set(split.train_images.tolist()).issubset({0, 1, 2, 3, 4, 5})
    assert set(split.tune_images.tolist()).issubset({0, 1, 2, 3, 4, 5})
    assert split.train_images.size == 3  # round(0.5 * 6)
    assert split.tune_images.size == 3

    other = build_reliability_split(eval_split, image_id, seed=8, train_frac=0.5)
    assert not np.array_equal(split.train_images, other.train_images)


# ---------------------------------------------------------------------------
# (b) row_mask semantics
# ---------------------------------------------------------------------------
def test_row_mask_partitions_val_calib_exactly():
    eval_split = np.array(
        ["val_calib", "val_calib", "val_calib", "testA", "testB", "val_select"],
        dtype="<U10",
    )
    image_id = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    split = ReliabilitySplit(
        seed=1,
        train_frac=0.5,
        train_images=np.array([0, 1], dtype=np.int64),
        tune_images=np.array([2], dtype=np.int64),
    )

    train_mask = split.row_mask(eval_split, image_id, kind="reliability_train")
    tune_mask = split.row_mask(eval_split, image_id, kind="reliability_tune")

    assert not np.any(train_mask & tune_mask)  # mutually exclusive
    val_rows = eval_split == data.DEFAULT_VAL_SPLIT_NAME
    assert np.array_equal(train_mask | tune_mask, val_rows)  # exhaustive over val_calib
    assert np.array_equal(train_mask, np.array([True, True, False, False, False, False]))
    assert np.array_equal(tune_mask, np.array([False, False, True, False, False, False]))

    with pytest.raises(ValueError):
        split.row_mask(eval_split, image_id, kind="testA")


# ---------------------------------------------------------------------------
# (c) manifest writer
# ---------------------------------------------------------------------------
def test_write_split_manifest_is_readable_stable_and_consistent(tmp_path):
    eval_split = np.array(["val_calib"] * 10, dtype="<U10")
    image_id = np.arange(10, dtype=np.int64)
    split = build_reliability_split(eval_split, image_id, seed=123, train_frac=0.7)

    path = tmp_path / "split_manifest.json"
    payload = write_split_manifest(path, split, extra={"note": "phase0.5"})

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert payload["note"] == "phase0.5"
    assert payload["n_images"] == 10
    assert payload["n_train"] + payload["n_tune"] == 10
    assert payload["train_images"] == [int(x) for x in split.train_images]
    assert payload["tune_images"] == [int(x) for x in split.tune_images]
    assert payload["sha256_train"] == _sha(split.train_images)
    assert payload["sha256_tune"] == _sha(split.tune_images)

    # deterministic hashes across re-writes (created_utc may move)
    again = write_split_manifest(path, split, extra={"note": "phase0.5"})
    assert again["sha256_train"] == payload["sha256_train"]
    assert again["sha256_tune"] == payload["sha256_tune"]
    assert json.loads(path.read_text(encoding="utf-8")) == again


# ---------------------------------------------------------------------------
# (d) synthetic loaders
# ---------------------------------------------------------------------------
def test_load_b3_style_npz(tmp_path):
    scores = np.array(
        [
            [5.0, 1.0, 0.0, 0.0, 0.0],
            [5.0, 4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 5.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [9.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0, 0.0],
        ]
    )
    # argmax = [0, 0, 4, 4, 0, 1]; rows 1, 3, 4, 5 are wrong
    target_local = np.array([0, 1, 4, 0, 1, 0], dtype=np.int32)
    eval_split = np.array(
        ["val_calib", "val_calib", "val_select", "testA", "testB", "val_calib"],
        dtype="<U10",
    )
    _write_b3(
        tmp_path,
        "seed_1",
        5,
        scores=scores,
        sentence_id=np.arange(100, 106),
        ref_id=np.arange(200, 206),
        image_id=np.array([0, 0, 1, 2, 3, 3]),
        eval_split=eval_split,
        target_local=target_local,
    )

    loaded = load_scorer_scores("b3_seed1", 5, b3_root=tmp_path, cosine_root=tmp_path)
    assert loaded.scorer_id == "b3_seed1"
    assert loaded.k == 5
    assert loaded.scores.shape == (6, 5)
    assert loaded.scores.dtype == np.float32
    assert loaded.target_local.dtype == np.int32
    assert np.array_equal(loaded.eval_split, eval_split)
    assert np.array_equal(loaded.sentence_id, np.arange(100, 106))
    assert np.array_equal(loaded.correct, np.array([1, 0, 1, 0, 0, 0], dtype=bool))
    assert set(np.unique(loaded.eval_split).tolist()) == {
        "val_calib",
        "val_select",
        "testA",
        "testB",
    }


def test_load_cosine_style_npz_filters_and_decodes(tmp_path):
    raw_scores = np.array(
        [
            [5.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 5.0],
            [3.0, 1.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    # codes follow SPLIT_CODES: val_select=0, val_calib=1, testA=2, testB=3
    split_codes = np.array([0, 1, 2, 3, 1, 0], dtype=np.int8)
    # argmax = [0, 1, 3, 4, 0, 0]; rows 2 and 3 of the kept slice are wrong
    target_local = np.array([0, 1, 4, 0, 0, 0], dtype=np.int32)
    _write_cosine(
        tmp_path,
        5,
        raw_scores=raw_scores,
        sentence_ids=np.array([10, 11, 12, 13, 14, 15]),
        ref_ids=np.array([20, 21, 22, 23, 24, 25]),
        image_ids=np.array([0, 1, 2, 3, 4, 5]),
        split_codes=split_codes,
        target_local=target_local,
    )

    common_ids = np.array([10, 11, 12, 13, 14], dtype=np.int64)  # drops sentence 15
    loaded = load_scorer_scores(
        "cosine", 5, cosine_root=tmp_path, common_ids=common_ids
    )
    assert loaded.scorer_id == "cosine"
    assert len(loaded) == 5
    assert np.array_equal(loaded.sentence_id, np.array([10, 11, 12, 13, 14]))
    assert np.array_equal(
        loaded.eval_split,
        np.array(["val_select", "val_calib", "testA", "testB", "val_calib"]),
    )
    assert set(np.unique(loaded.eval_split).tolist()) == {
        "val_select",
        "val_calib",
        "testA",
        "testB",
    }
    assert np.array_equal(loaded.correct, np.array([1, 1, 0, 0, 1], dtype=bool))

    # a missing common-cohort row raises with the missing count
    with pytest.raises(ValueError, match="missing"):
        load_scorer_scores(
            "cosine", 5, cosine_root=tmp_path, common_ids=np.array([11, 12, 999])
        )


def test_scorer_ids_frozen():
    assert SCORER_IDS == ("b3_seed1", "b3_seed2", "b3_seed3", "cosine")


# ---------------------------------------------------------------------------
# (e) real-data smoke test (skipped when the Phase 0 artifacts are absent)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not REAL_B3_K5.exists(), reason="real Phase 0 B3 artifacts not present")
def test_real_b3_and_cosine_common_cohort():
    seed1 = load_scorer_scores(
        "b3_seed1", 5, b3_root=REAL_B3_ROOT, cosine_root=REAL_COSINE_ROOT
    )
    assert len(seed1) == 20799

    # argmax == target_local is exactly per_sentence_predictions.npz's correct_K5
    with np.load(REAL_B3_ROOT / "seed_1" / "per_sentence_predictions.npz") as ref:
        assert np.array_equal(seed1.correct, ref["correct_K5"].astype(bool))
        assert np.array_equal(seed1.sentence_id, ref["sentence_ids"])

    common = common_sentence_ids(b3_root=REAL_B3_ROOT)
    cosine = load_scorer_scores(
        "cosine", 5, b3_root=REAL_B3_ROOT, cosine_root=REAL_COSINE_ROOT, common_ids=common
    )
    assert len(cosine) == 20799
    assert np.array_equal(np.unique(cosine.sentence_id), common)
    assert np.array_equal(np.unique(cosine.sentence_id), np.unique(seed1.sentence_id))

    seed2 = load_scorer_scores(
        "b3_seed2", 5, b3_root=REAL_B3_ROOT, cosine_root=REAL_COSINE_ROOT
    )
    assert np.array_equal(seed1.sentence_id, seed2.sentence_id)
    assert not np.array_equal(seed1.scores, seed2.scores)
