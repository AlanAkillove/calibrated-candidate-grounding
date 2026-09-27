"""Unit tests for :mod:`ccg.experiment.phase0a` - pure NumPy, no real caches.

The scoring tests use tiny duck-typed caches (3 images x 6 proposals x 16-d
normalised vectors): ``score_sets`` only needs ``region_features(image_id)``
and ``text_features(sentence_id)``, so the 512-d storage contract of
:func:`ccg.features.cache.write_feature_cache` does not apply here (the real
cache is exercised end-to-end in ``test_phase0a_pipeline.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from ccg.experiment import phase0a
from ccg.experiment.phase0a import (
    CalibrationIsolationError,
    SampleStats,
    ScoreBundle,
    SentenceRecord,
    ValidationFailure,
    accuracy_metric,
    assert_accuracy_monotonic,
    assert_rank_monotonic,
    assert_score_invariance,
    compute_candidate_scores,
    ece_metric,
    fit_global_temperature,
    paired_cluster_bootstrap,
    ranking_metrics,
    score_sets,
    target_local_index,
    top1_delta_vs_k5,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _unit_rows(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    v = rng.normal(size=(n, d))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


class _TinyCache:
    """Duck-typed stand-in for FeatureCache (float16 storage, like the real one)."""

    def __init__(self, regions, texts):
        self._regions = {int(k): np.asarray(v, dtype=np.float16) for k, v in regions.items()}
        self._texts = {int(k): np.asarray(v, dtype=np.float16) for k, v in texts.items()}

    def region_features(self, image_id: int) -> np.ndarray:
        return self._regions[int(image_id)]

    def text_features(self, sentence_id: int) -> np.ndarray:
        return self._texts[int(sentence_id)]


def _record(sentence_id, ref_id, image_id, target_index, order, *, split="val_select"):
    return SentenceRecord(
        sentence_id=int(sentence_id),
        ref_id=int(ref_id),
        image_id=int(image_id),
        eval_split=str(split),
        target_index=None if target_index is None else int(target_index),
        distractor_order=np.asarray(order, dtype=np.int32),
        target_max_iou=0.9,
        eligible={5: True, 10: True},
    )


def _make_bundle(
    scores_by_k,
    local_by_k,
    *,
    image_ids,
    splits,
    sentence_ids=None,
    ref_ids=None,
) -> ScoreBundle:
    """ScoreBundle built directly from fixed scores (no cache, full control)."""
    ks = tuple(sorted(int(k) for k in scores_by_k))
    n = int(np.asarray(scores_by_k[ks[0]]).shape[0])
    if sentence_ids is None:
        sentence_ids = np.arange(n)
    if ref_ids is None:
        ref_ids = np.asarray(sentence_ids)
    records_by_k = {
        k: [
            _record(sentence_ids[i], ref_ids[i], image_ids[i], local_by_k[k][i], [], split=splits[i])
            for i in range(n)
        ]
        for k in ks
    }
    return ScoreBundle(
        ks=ks,
        records_by_k=records_by_k,
        raw_scores={k: np.asarray(scores_by_k[k], dtype=np.float32) for k in ks},
        candidate_indices={k: np.tile(np.arange(k, dtype=np.int32), (n, 1)) for k in ks},
        target_local={k: np.asarray(local_by_k[k], dtype=np.int32) for k in ks},
        n_skipped_target_missing={k: 0 for k in ks},
        n_skipped_insufficient={k: 0 for k in ks},
    )


# ---------------------------------------------------------------------------
# score_sets
# ---------------------------------------------------------------------------
def test_score_sets_assembles_target_first_and_matches_manual_scores():
    rng = np.random.default_rng(0)
    n_props, dim = 6, 16
    images = (10, 20, 30)
    regions = {image_id: _unit_rows(rng, n_props, dim) for image_id in images}
    texts = {sid: _unit_rows(rng, 1, dim)[0] for sid in range(6)}
    cache = _TinyCache(regions, texts)

    records = []
    for ref_pos, image_id in enumerate(images):
        target = ref_pos + 1  # bank row
        order = [i for i in range(n_props) if i != target][:5]
        for s in range(2):
            records.append(
                _record(2 * ref_pos + s, ref_pos + 1, image_id, target, order)
            )

    bundle = score_sets(cache, records, ks=(3, 5))
    assert bundle.ks == (3, 5)
    assert bundle.n_skipped_target_missing == {3: 0, 5: 0}
    assert bundle.n_skipped_insufficient == {3: 0, 5: 0}

    for k in (3, 5):
        cand = bundle.candidate_indices[k]
        assert cand.shape == (6, k)
        for row, rec in enumerate(bundle.records_by_k[k]):
            expected = np.concatenate(
                [[rec.target_index], np.asarray(rec.distractor_order, dtype=np.int64)[: k - 1]]
            )
            assert np.array_equal(cand[row].astype(np.int64), expected), (k, row)
        # the local index is *searched*; here it happens to be 0 - verify by value
        assert bundle.target_local[k].shape == (6,)
        assert np.array_equal(
            bundle.target_local[k], np.zeros(6, dtype=np.int32)
        ), bundle.target_local[k]

        # manual scores: replicate the FP16 storage -> FP32 -> normalise path
        # and the *full-matrix* gemv + gather form of the default hook
        for row, rec in enumerate(bundle.records_by_k[k]):
            zq = np.asarray(texts[rec.sentence_id], dtype=np.float16).astype(np.float32)
            zq = zq / np.float32(np.linalg.norm(zq))
            zi = np.asarray(regions[rec.image_id], dtype=np.float16).astype(np.float32)
            zi = zi / np.linalg.norm(zi, axis=1, keepdims=True).astype(np.float32)
            expected = (zi @ zq)[cand[row]].astype(np.float32)
            assert np.allclose(bundle.raw_scores[k][row], expected, rtol=1e-6, atol=1e-7)


def test_score_sets_shared_candidates_are_bit_identical_across_ks():
    rng = np.random.default_rng(1)
    regions = {1: _unit_rows(rng, 8, 16)}
    texts = {sid: _unit_rows(rng, 1, 16)[0] for sid in range(3)}
    cache = _TinyCache(regions, texts)
    order = [0, 2, 4, 5, 7, 1, 3, 6]
    records = [_record(sid, 1, 1, 1, order) for sid in range(3)]

    bundle = score_sets(cache, records, ks=(3, 5, 20))
    # K=20 needs order[:19] which does not exist -> every row skipped for K=20
    assert bundle.raw_scores[20].shape == (0, 20)
    assert bundle.n_skipped_insufficient[20] == 3

    for row in range(3):
        # C_3 = [target] + order[:2]; C_5 = [target] + order[:4]: prefix shared
        assert np.array_equal(bundle.raw_scores[3][row], bundle.raw_scores[5][row][:3])
    # and the checker itself passes on this bundle
    summary = assert_score_invariance(bundle)
    assert summary["status"] == "passed"
    assert summary["n_violations"] == 0


def test_score_sets_renormalises_vectors_and_counts_out_of_tolerance():
    rng = np.random.default_rng(2)
    good = _unit_rows(rng, 4, 16)
    bad = good * np.float32(2.0)  # stored norm 2 -> out of tolerance, renormalised
    cache = _TinyCache({1: np.vstack([good, bad])}, {0: _unit_rows(rng, 1, 16)[0]})
    records = [_record(0, 1, 1, 0, [1, 2, 3])]

    bundle = score_sets(cache, records, ks=(4,))
    assert bundle.n_l2_renormalized >= 4  # the four norm-2 rows
    assert np.all(np.isfinite(bundle.raw_scores[4]))


def test_score_sets_raises_on_zero_norm_vector():
    rng = np.random.default_rng(3)
    region = _unit_rows(rng, 4, 16)
    region[2] = 0.0
    cache = _TinyCache({1: region}, {0: _unit_rows(rng, 1, 16)[0]})
    with pytest.raises(ValueError, match="zero-norm"):
        score_sets(cache, [_record(0, 1, 1, 0, [1, 2, 3])], ks=(4,))


def test_score_sets_skips_missing_targets_and_reports_counts():
    rng = np.random.default_rng(4)
    cache = _TinyCache({1: _unit_rows(rng, 4, 16)}, {0: _unit_rows(rng, 1, 16)[0]})
    records = [
        _record(0, 1, 1, None, [1, 2, 3]),  # natural miss -> skipped everywhere
        _record(1, 2, 1, 0, [1, 2]),  # ordering too short for K=4
    ]
    bundle = score_sets(cache, records, ks=(4,))
    assert bundle.raw_scores[4].shape == (0, 4)
    assert bundle.n_skipped_target_missing[4] == 1
    assert bundle.n_skipped_insufficient[4] == 1


# ---------------------------------------------------------------------------
# target_local_index
# ---------------------------------------------------------------------------
def test_target_local_index_searches_by_value():
    assert target_local_index(np.array([7, 3, 9]), 9) == 2
    assert target_local_index(np.array([7, 3, 9]), 7) == 0
    with pytest.raises(ValueError, match="not found"):
        target_local_index(np.array([7, 3, 9]), 5)
    batch = target_local_index(np.array([[1, 2, 3], [4, 3, 2]]), np.array([3, 4]))
    assert np.array_equal(batch, np.array([2, 0], dtype=np.int64))
    with pytest.raises(ValueError, match="not found"):
        target_local_index(np.array([[1, 2, 3]]), np.array([9]))


def test_compute_candidate_scores_gathers_from_one_full_gemv():
    rng = np.random.default_rng(5)
    rows = _unit_rows(rng, 10, 16)
    query = _unit_rows(rng, 1, 16)[0]
    cand = np.array([4, 0, 7], dtype=np.int64)
    scores = compute_candidate_scores(query, rows, cand, K=3)
    expected = (rows @ query)[cand]
    assert np.array_equal(scores, expected.astype(np.float32))


# ---------------------------------------------------------------------------
# ranking metrics: DeltaAcc / mean rank / MRR hand-checked
# ---------------------------------------------------------------------------
def test_ranking_metrics_and_delta_acc_match_hand_computed_values():
    scores5 = np.array(
        [
            [0.10, 0.50, 0.30, 0.20, 0.40],  # s0: target local 1 -> correct, rank 1
            [0.60, 0.20, 0.70, 0.30, 0.10],  # s1: target 0.6, one higher -> rank 2
            [0.20, 0.30, 0.10, 0.40, 0.90],  # s2: target 0.4, one higher -> rank 2
            [0.30, 0.30, 0.30, 0.30, 0.30],  # s3: all ties -> rank 1, argmax 0
        ]
    )
    local5 = np.array([1, 0, 3, 2])
    scores10 = np.column_stack(
        [
            scores5,
            np.array(
                [
                    [0.99, -1.0, -1.0, -1.0, -1.0],  # s0 flips to wrong
                    [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0],
                    [-1.0, -1.0, -1.0, -1.0, -1.0],
                ]
            ),
        ]
    )
    local10 = local5.copy()
    image_ids = np.array([1, 1, 1, 2])
    splits = np.array(["val_select"] * 4)
    bundle = _make_bundle(
        {5: scores5, 10: scores10},
        {5: local5, 10: local10},
        image_ids=image_ids,
        splits=splits,
    )

    rows = ranking_metrics(bundle)
    cells = {(r["eval_split"], r["K"], r["denominator"]): r for r in rows}

    r5 = cells[("val_select", 5, "common")]
    assert r5["n"] == 4
    assert r5["top1"] == pytest.approx(0.25)
    assert r5["top5"] == pytest.approx(1.0)
    # ranks [1, 2, 2, 3]: the s3 ties sort by ascending index, so rank 3
    assert r5["mean_target_rank"] == pytest.approx(2.0)
    assert r5["mrr"] == pytest.approx((1.0 + 0.5 + 0.5 + 1.0 / 3.0) / 4.0)
    assert r5["delta_acc_vs_k5"] == pytest.approx(0.0)

    r10 = cells[("val_select", 10, "common")]
    assert r10["top1"] == pytest.approx(0.0)
    assert r10["top5"] == pytest.approx(1.0)
    # ranks [2, 2, 2, 3]
    assert r10["mean_target_rank"] == pytest.approx(2.25)
    assert r10["mrr"] == pytest.approx((0.5 + 0.5 + 0.5 + 1.0 / 3.0) / 4.0)
    assert r10["delta_acc_vs_k5"] == pytest.approx(-0.25)

    delta = top1_delta_vs_k5(bundle)
    assert delta["val_select"][5] == pytest.approx(0.0)
    assert delta["val_select"][10] == pytest.approx(-0.25)
    assert delta["__pooled__"][10] == pytest.approx(-0.25)


# ---------------------------------------------------------------------------
# calibration metrics: adaptive ECE known values
# ---------------------------------------------------------------------------
def test_ece_metric_perfect_calibration_is_zero():
    # two confidence blocks that the adaptive binner cannot split (ties snapped):
    # block 0.25 -> 8/32 correct, block 0.75 -> 21/28 correct
    conf = np.concatenate([np.full(32, 0.25), np.full(28, 0.75)])
    correct = np.concatenate([np.array([1.0] * 8 + [0.0] * 24), np.array([1.0] * 21 + [0.0] * 7)])
    stats = SampleStats.from_conf_correct(conf, correct)
    assert ece_metric(stats) == pytest.approx(0.0, abs=1e-12)


def test_ece_metric_overconfident_is_known_gap():
    conf = np.ones(40)
    correct = np.concatenate([np.ones(20), np.zeros(20)])
    stats = SampleStats.from_conf_correct(conf, correct)
    assert ece_metric(stats) == pytest.approx(0.5)
    assert accuracy_metric(stats) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# hard sanity checkers
# ---------------------------------------------------------------------------
def _nested_pair(n=2, *, mutate_hi=False):
    """K=5 / K=10 bundle with bit-identical shared columns (unless mutated)."""
    rng = np.random.default_rng(7)
    base = rng.uniform(0.0, 0.5, size=(n, 5)).astype(np.float32)
    local5 = np.array([1, 0])[:n]
    for row in range(n):
        base[row, local5[row]] = 0.9  # target clearly wins at K=5
    hi = np.column_stack([base, np.full((n, 5), -1.0, dtype=np.float32)])
    if mutate_hi:
        hi[0, 0] = np.float32(hi[0, 0] + np.float32(0.5))  # shared candidate drifts
    return _make_bundle(
        {5: base, 10: hi},
        {5: local5, 10: local5},
        image_ids=np.arange(n),
        splits=np.array(["val_select"] * n),
    )


def test_score_invariance_checker_passes_and_raises():
    good = _nested_pair()
    summary = assert_score_invariance(good)
    assert summary["status"] == "passed"
    assert summary["n_candidate_pairs"] == 10
    assert summary["max_abs_diff"] == 0.0

    bad = _nested_pair(mutate_hi=True)
    with pytest.raises(ValidationFailure) as excinfo:
        assert_score_invariance(bad)
    assert excinfo.value.check == "score_invariance"
    assert excinfo.value.violations
    assert excinfo.value.violations[0]["candidate"] == 0


def test_rank_monotonic_checker_passes_and_raises():
    # legal: ranks 2 then 2 (adding low-score distractors never helps the target)
    scores5 = np.array([[0.60, 0.20, 0.70, 0.30, 0.10]], dtype=np.float32)
    scores10 = np.column_stack([scores5, np.full((1, 5), -1.0, dtype=np.float32)])
    local = np.array([0])
    good = _make_bundle(
        {5: scores5, 10: scores10},
        {5: local, 10: local},
        image_ids=np.array([1]),
        splits=np.array(["val_select"]),
    )
    summary = assert_rank_monotonic(good)
    assert summary["status"] == "passed"
    assert summary["n_violations"] == 0

    # illegal: K=10 demotes the only higher-scoring distractor -> rank 1 < 2
    scores10_bad = scores10.copy()
    scores10_bad[0, 2] = np.float32(0.05)
    bad = _make_bundle(
        {5: scores5, 10: scores10_bad},
        {5: local, 10: local},
        image_ids=np.array([1]),
        splits=np.array(["val_select"]),
    )
    with pytest.raises(ValidationFailure) as excinfo:
        assert_rank_monotonic(bad)
    assert excinfo.value.check == "rank_monotonic"
    assert excinfo.value.violations[0]["rank_lo"] == 2
    assert excinfo.value.violations[0]["rank_hi"] == 1


def test_accuracy_monotonic_checker_passes_and_raises():
    # legal: 1/2 correct at K=5, 0/2 at K=10
    scores5 = np.array(
        [[0.10, 0.90, 0.30, 0.20, 0.40], [0.10, 0.20, 0.30, 0.40, 0.90]], dtype=np.float32
    )
    local = np.array([1, 1])
    scores10 = np.column_stack([scores5, np.full((2, 5), -1.0, dtype=np.float32)])
    scores10[0, 0] = np.float32(0.99)  # s0 flips to wrong at K=10
    good = _make_bundle(
        {5: scores5, 10: scores10},
        {5: local, 10: local},
        image_ids=np.array([1, 1]),
        splits=np.array(["val_select"] * 2),
    )
    summary = assert_accuracy_monotonic(good)
    assert summary["status"] == "passed"
    assert summary["accuracies"]["5"] == pytest.approx(0.5)
    assert summary["accuracies"]["10"] == pytest.approx(0.0)

    # illegal: accuracy *gains* with more distractors (variance, not monotonicity)
    bad = _make_bundle(
        {5: scores10, 10: scores5},  # swapped: 0.0 -> 0.5
        {5: local, 10: local},
        image_ids=np.array([1, 1]),
        splits=np.array(["val_select"] * 2),
    )
    with pytest.raises(ValidationFailure) as excinfo:
        assert_accuracy_monotonic(bad)
    assert excinfo.value.check == "accuracy_monotonic"


# ---------------------------------------------------------------------------
# paired cluster bootstrap
# ---------------------------------------------------------------------------
def test_paired_cluster_bootstrap_seed_reproducible_and_ci_covers_diff():
    rng = np.random.default_rng(11)
    n = 600
    conf = rng.uniform(0.5, 1.0, n)
    correct_a = np.concatenate([np.ones(480), np.zeros(120)])
    correct_b = np.concatenate([np.ones(360), np.zeros(240)])
    a = SampleStats.from_conf_correct(conf, correct_a)
    b = SampleStats.from_conf_correct(conf, correct_b)

    r1 = paired_cluster_bootstrap(accuracy_metric, a, b, None, n_replicates=1000, seed=0)
    r2 = paired_cluster_bootstrap(accuracy_metric, a, b, None, n_replicates=1000, seed=0)
    assert np.array_equal(r1["replicates"], r2["replicates"])
    r3 = paired_cluster_bootstrap(accuracy_metric, a, b, None, n_replicates=1000, seed=1)
    assert not np.array_equal(r1["replicates"], r3["replicates"])

    assert r1["diff"] == pytest.approx(0.2)
    assert r1["ci_low"] < 0.2 < r1["ci_high"]
    assert r1["resample_unit"] == "sample"
    assert r1["n_replicates"] == 1000


def test_paired_cluster_bootstrap_clusters_widen_ci():
    rng = np.random.default_rng(12)
    n_clusters, per_cluster = 30, 20
    n = n_clusters * per_cluster
    cluster_ids = np.repeat(np.arange(n_clusters), per_cluster)
    # strong within-image correlation, anti-correlated pair: every image is a
    # single Bernoulli draw, so the honest interval must be far wider
    a_cluster = np.concatenate([np.ones(15), np.zeros(15)])
    b_cluster = np.concatenate([np.zeros(15), np.ones(15)])
    correct_a = np.repeat(a_cluster, per_cluster)
    correct_b = np.repeat(b_cluster, per_cluster)
    conf = rng.uniform(0.4, 0.9, n)
    a = SampleStats.from_conf_correct(conf, correct_a)
    b = SampleStats.from_conf_correct(conf, correct_b)

    sample_level = paired_cluster_bootstrap(accuracy_metric, a, b, None, n_replicates=2000, seed=0)
    cluster_level = paired_cluster_bootstrap(
        accuracy_metric, a, b, cluster_ids, n_replicates=2000, seed=0
    )
    assert cluster_level["n_clusters"] == n_clusters
    assert cluster_level["resample_unit"] == "image"
    width_sample = sample_level["ci_high"] - sample_level["ci_low"]
    width_cluster = cluster_level["ci_high"] - cluster_level["ci_low"]
    assert width_cluster > 2.0 * width_sample, (width_cluster, width_sample)


def test_paired_cluster_bootstrap_validates_inputs():
    stats = SampleStats.from_conf_correct(np.full(4, 0.5), np.array([1.0, 0.0, 1.0, 0.0]))
    other = SampleStats.from_conf_correct(np.full(3, 0.5), np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError, match="rows"):
        paired_cluster_bootstrap(accuracy_metric, stats, other, None)
    with pytest.raises(ValueError, match="cluster_ids"):
        paired_cluster_bootstrap(accuracy_metric, stats, stats, np.array([1, 2]))


# ---------------------------------------------------------------------------
# calibration isolation (defensive tests)
# ---------------------------------------------------------------------------
def _isolation_bundle():
    """4 rows: two val_select (0, 1) and two val_calib (2, 3), K=5 and K=10."""
    rng = np.random.default_rng(21)
    scores5 = rng.uniform(0.0, 1.0, size=(4, 5)).astype(np.float32)
    scores10 = np.column_stack([scores5, rng.uniform(-1.0, 0.0, size=(4, 5)).astype(np.float32)])
    local = np.array([1, 0, 2, 1])
    return _make_bundle(
        {5: scores5, 10: scores10},
        {5: local, 10: local},
        image_ids=np.array([1, 2, 3, 4]),
        splits=np.array(["val_select", "val_select", "val_calib", "val_calib"]),
        sentence_ids=np.arange(4),
    )


def test_fit_global_temperature_rejects_non_val_calib_rows():
    bundle = _isolation_bundle()
    # default: every row enters -> val_select rows are an isolation violation
    with pytest.raises(CalibrationIsolationError, match="val_calib"):
        fit_global_temperature(bundle, ks=(5, 10))
    # explicit rows that include val_select rows are equally rejected
    with pytest.raises(CalibrationIsolationError):
        fit_global_temperature(
            bundle,
            ks=(5,),
            rows_by_k={5: np.array([0, 1, 2, 3], dtype=np.int64)},
        )
    # same for a testA row smuggled into the fit
    bundle_testa = _make_bundle(
        {5: bundle.raw_scores[5], 10: bundle.raw_scores[10]},
        {5: bundle.target_local[5], 10: bundle.target_local[10]},
        image_ids=bundle.image_ids(5),
        splits=np.array(["testA", "testA", "val_calib", "val_calib"]),
        sentence_ids=np.arange(4),
    )
    with pytest.raises(CalibrationIsolationError):
        fit_global_temperature(bundle_testa, ks=(5,))


def test_fit_global_temperature_accepts_val_calib_only_rows():
    bundle = _isolation_bundle()
    fit = fit_global_temperature(
        bundle,
        ks=(5, 10),
        rows_by_k={5: np.array([2, 3], dtype=np.int64), 10: np.array([2, 3], dtype=np.int64)},
    )
    assert fit.split == "val_calib"
    assert fit.kind == "global"
    assert fit.ks == (5, 10)
    assert fit.n_sets == 4
    assert fit.temperature > 0.0
    assert fit.nll_after <= fit.nll_before + 1e-9


def test_fit_global_temperature_rejects_illegal_ks():
    bundle = _isolation_bundle()
    with pytest.raises(ValueError, match="oracle"):
        # K=20 fitting is the (illegal) oracle diagnostic, not a legal fit
        fit_global_temperature(bundle, ks=(5, 20))
