"""End-to-end smoke test for the Phase 0.5 reliability data + evaluate pipeline.

A tiny synthetic B3-style corpus (40 images x 5 sentences = 200 rows, four
splits, ``K in {5, 10, 20, 50}``) is written to ``tmp_path``.  The test then runs
the frozen flow: ``common_sentence_ids`` -> ``load_scorer_scores`` ->
``build_reliability_split`` -> ``write_split_manifest`` -> point metrics on
``testA`` -> a short image-clustered cross-``K`` bootstrap.

The reliability-model half (``ccg.reliability.models``) is written by a parallel
agent; it is exercised only when importable and exposing a compatible
``LogisticModel`` API, otherwise the test skips.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ccg.metrics.calibration import confidence_from_scores
from ccg.metrics.discrimination import auroc_correct
from ccg.reliability import data
from ccg.reliability.evaluate import cross_k_bootstrap_row, point_metric_row

#: split layout of the synthetic corpus: (eval split name, number of images).
_LAYOUT: tuple[tuple[str, int], ...] = (
    ("val_calib", 20),
    ("val_select", 5),
    ("testA", 7),
    ("testB", 8),
)
_SENTENCES_PER_IMAGE = 5
_KS: tuple[int, ...] = (5, 10, 20, 50)


# ---------------------------------------------------------------------------
# synthetic corpus
# ---------------------------------------------------------------------------
def _build_corpus(root: Path) -> None:
    """Write a deterministic B3-style corpus; every image has 3 correct + 2 wrong rows."""
    image_ids: list[int] = []
    splits: list[str] = []
    sentence_ids: list[int] = []
    next_sentence = 0
    for name, count in _LAYOUT:
        for image in range(count):
            for _ in range(_SENTENCES_PER_IMAGE):
                image_ids.append(image)
                splits.append(name)
                sentence_ids.append(next_sentence)
                next_sentence += 1

    image_id = np.asarray(image_ids, dtype=np.int64)
    eval_split = np.asarray(splits, dtype="<U10")
    sentence_id = np.asarray(sentence_ids, dtype=np.int64)
    ref_id = sentence_id.copy()
    n = sentence_id.size

    rng = np.random.default_rng(20260928)
    # per-image position 0..4: positions 0-2 are "correct" (target 0 wins).
    position = np.tile(np.arange(_SENTENCES_PER_IMAGE), len(image_id) // _SENTENCES_PER_IMAGE)
    win = position < 3
    target_local = np.zeros(n, dtype=np.int32)
    for k in _KS:
        scores = rng.normal(size=(n, k)).astype(np.float32)
        scores[win, 0] += 6.0
        lose = np.nonzero(~win)[0]
        other = rng.integers(1, k, size=lose.size)
        scores[lose, other] += 6.0
        out = root / "seed_1" / "raw_scores"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / f"K{k}.npz",
            scores=scores,
            sentence_id=sentence_id,
            ref_id=ref_id,
            image_id=image_id,
            eval_split=eval_split,
            target_local=target_local,
        )


def _features(scores: np.ndarray) -> np.ndarray:
    """Minimal hand-written per-row score features (fallback when features.py is absent)."""
    msp, _ = confidence_from_scores(scores)
    ordered = np.sort(scores, axis=1)
    top1 = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    return np.column_stack([msp, top1, margin]).astype(np.float64)


def _feature_matrix(scores: np.ndarray, mask: np.ndarray, features_module) -> np.ndarray:
    """L1 stat features when ``features.py`` is ready, else the minimal fallback."""
    if features_module is not None and hasattr(features_module, "stat_features"):
        return np.asarray(
            features_module.stat_features(scores[mask], temperature=1.0), dtype=np.float64
        )
    return _features(scores[mask])


def _predict_scores(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)).reshape(-1)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(features)).reshape(-1)
    raise TypeError("model exposes neither predict_proba nor predict")


# ---------------------------------------------------------------------------
# data -> evaluate end-to-end
# ---------------------------------------------------------------------------
def test_pipeline_data_and_evaluate_end_to_end(tmp_path):
    _build_corpus(tmp_path)

    common = data.common_sentence_ids(b3_root=tmp_path)
    assert common.size == 200

    scores5 = data.load_scorer_scores(
        "b3_seed1", 5, b3_root=tmp_path, cosine_root=tmp_path
    )
    assert len(scores5) == 200

    # image-level split over the 20 val_calib images (0.70 -> 14 / 6)
    split = data.build_reliability_split(scores5.eval_split, scores5.image_id)
    assert split.train_images.size == 14
    assert split.tune_images.size == 6

    manifest_path = tmp_path / "split_manifest.json"
    payload = data.write_split_manifest(manifest_path, split)
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == payload
    joined = ",".join(str(int(x)) for x in split.train_images)
    assert payload["sha256_train"] == hashlib.sha256(joined.encode("utf-8")).hexdigest()

    train_mask = split.row_mask(scores5.eval_split, scores5.image_id, kind="reliability_train")
    tune_mask = split.row_mask(scores5.eval_split, scores5.image_id, kind="reliability_tune")
    assert train_mask.sum() == 14 * _SENTENCES_PER_IMAGE
    assert tune_mask.sum() == 6 * _SENTENCES_PER_IMAGE
    assert not np.any(train_mask & tune_mask)

    # testA point metrics at K5 and K50 (confidence = max softmax probability)
    scores50 = data.load_scorer_scores(
        "b3_seed1", 50, b3_root=tmp_path, cosine_root=tmp_path
    )
    test_mask = scores5.eval_split == "testA"
    assert test_mask.sum() == 7 * _SENTENCES_PER_IMAGE

    conf5, _ = confidence_from_scores(scores5.scores[test_mask])
    corr5 = scores5.correct[test_mask].astype(np.float64)
    conf50, _ = confidence_from_scores(scores50.scores[test_mask])
    corr50 = scores50.correct[test_mask].astype(np.float64)

    row5 = point_metric_row(conf5, corr5)
    row50 = point_metric_row(conf50, corr50)
    for row in (row5, row50):
        assert np.isfinite(row["auroc_correct"])
        assert np.isfinite(row["e_aurc"])
        assert np.isfinite(row["rer_at_50"])

    clusters = scores5.image_id[test_mask]
    rows = cross_k_bootstrap_row(
        conf5,
        corr5,
        conf50,
        corr50,
        clusters,
        eval_split="testA",
        k_lo=5,
        k_hi=50,
        replicates=50,
        seed=0,
    )
    assert len(rows) == 5  # 4 absolute + 1 relative e_aurc
    for row in rows:
        assert row["resample_unit"] == "image"
        assert row["n_clusters"] == 7
        assert np.isfinite(row["diff"])
        assert np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"])


# ---------------------------------------------------------------------------
# optional reliability-model half (parallel dependency; skips until available)
# ---------------------------------------------------------------------------
def test_pipeline_reliability_model_integration(tmp_path):
    models = pytest.importorskip("ccg.reliability.models")
    if not hasattr(models, "LogisticModel"):
        pytest.skip("ccg.reliability.models.LogisticModel not available yet")
    try:
        from ccg.reliability import features as features_module
    except Exception:  # pragma: no cover - features.py landed by a parallel agent
        features_module = None

    _build_corpus(tmp_path)
    scores5 = data.load_scorer_scores(
        "b3_seed1", 5, b3_root=tmp_path, cosine_root=tmp_path
    )
    scores10 = data.load_scorer_scores(
        "b3_seed1", 10, b3_root=tmp_path, cosine_root=tmp_path
    )
    scores50 = data.load_scorer_scores(
        "b3_seed1", 50, b3_root=tmp_path, cosine_root=tmp_path
    )
    split = data.build_reliability_split(scores5.eval_split, scores5.image_id)
    train_mask = split.row_mask(scores5.eval_split, scores5.image_id, kind="reliability_train")
    tune_mask = split.row_mask(scores5.eval_split, scores5.image_id, kind="reliability_tune")

    # reliability_train rows at K in {5, 10} (protocol A6.2)
    x_train = np.vstack(
        [
            _feature_matrix(scores5.scores, train_mask, features_module),
            _feature_matrix(scores10.scores, train_mask, features_module),
        ]
    )
    y_train = np.concatenate(
        [scores5.correct[train_mask], scores10.correct[train_mask]]
    ).astype(np.float64)

    # reliability_train mu/sigma shared by every row (protocol A6.3)
    mu = x_train.mean(axis=0)
    sigma = x_train.std(axis=0) + 1e-8
    x_train = (x_train - mu) / sigma

    try:
        model = models.LogisticModel()
        model.fit(x_train, y_train)
    except (AttributeError, TypeError, NotImplementedError) as exc:  # pragma: no cover
        pytest.skip(f"LogisticModel API not compatible yet: {exc}")

    # model selection on reliability_tune: mean AUROC over K5 / K10 (A6.4)
    tune_aurocs = []
    for scores in (scores5, scores10):
        x_tune = (_feature_matrix(scores.scores, tune_mask, features_module) - mu) / sigma
        prob = _predict_scores(model, x_tune)
        tune_aurocs.append(auroc_correct(prob, scores.correct[tune_mask].astype(np.float64)))
    mean_auroc = float(np.mean(tune_aurocs))
    assert np.isfinite(mean_auroc)

    # frozen test evaluation: testA K5 vs K50 (point + short cross-K bootstrap)
    test_mask = scores5.eval_split == "testA"
    clusters = scores5.image_id[test_mask]
    corr5 = scores5.correct[test_mask].astype(np.float64)
    corr50 = scores50.correct[test_mask].astype(np.float64)
    prob5 = _predict_scores(
        model, (_feature_matrix(scores5.scores, test_mask, features_module) - mu) / sigma
    )
    prob50 = _predict_scores(
        model, (_feature_matrix(scores50.scores, test_mask, features_module) - mu) / sigma
    )
    row5 = point_metric_row(prob5, corr5, probability=prob5)
    row50 = point_metric_row(prob50, corr50, probability=prob50)
    assert np.isfinite(row5["auroc_correct"]) and np.isfinite(row50["auroc_correct"])
    assert np.isfinite(row5["ece_adaptive"])

    rows = cross_k_bootstrap_row(
        prob5,
        corr5,
        prob50,
        corr50,
        clusters,
        eval_split="testA",
        k_lo=5,
        k_hi=50,
        replicates=50,
        seed=0,
    )
    assert len(rows) == 5
    for row in rows:
        assert row["resample_unit"] == "image"
        assert np.isfinite(row["diff"])
    assert all(np.isfinite(r["worsening"]) for r in rows if r["diff_kind"] == "relative")
