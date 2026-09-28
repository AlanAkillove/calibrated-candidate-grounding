"""Unit tests for :mod:`ccg.reliability.evaluate` - pure NumPy, no real caches.

Covers the point metrics (perfect / random rankings and a hand-computed RER),
the image-clustered paired bootstrap rows (absolute + relative ``e_aurc``), the
phase0b ratio -> unified worsening mapping (the sign trap of the previous phase),
the sufficiency gate and the model-vs-model row schema.
"""

from __future__ import annotations

import numpy as np
import pytest

from ccg.metrics.selective import rer_at_coverage
from ccg.reliability import evaluate
from ccg.reliability.evaluate import (
    cross_k_bootstrap_row,
    model_vs_model_bootstrap_row,
    point_metric_row,
    sufficiency_verdict,
    worsening_ci_from_ratio_ci,
    worsening_from_ratio,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cell(
    split,
    k,
    *,
    auroc_drop=0.0,
    auroc_lo=0.0,
    auroc_hi=0.0,
    e_worsen=0.0,
    e_lo=0.0,
    e_hi=0.0,
    rer_drop=0.0,
    rer_lo=0.0,
    rer_hi=0.0,
    e_abs=0.01,
):
    return {
        "split": split,
        "K": k,
        "auroc_drop": auroc_drop,
        "auroc_drop_ci_low": auroc_lo,
        "auroc_drop_ci_high": auroc_hi,
        "e_aurc_worsening": e_worsen,
        "e_aurc_ci_low": e_lo,
        "e_aurc_ci_high": e_hi,
        "rer50_drop": rer_drop,
        "rer50_ci_low": rer_lo,
        "rer50_ci_high": rer_hi,
        "e_aurc_abs": e_abs,
    }


ABS_KEYS = {
    "eval_split",
    "K_lo",
    "K_hi",
    "metric",
    "diff_kind",
    "diff",
    "ci_low",
    "ci_high",
    "mean_lo",
    "mean_hi",
    "n",
    "n_clusters",
    "resample_unit",
    "n_replicates",
    "ci_level",
}

MODEL_KEYS = {
    "eval_split",
    "K",
    "model_a",
    "model_b",
    "metric",
    "diff",
    "ci_low",
    "ci_high",
    "mean_a",
    "mean_b",
    "n",
    "n_clusters",
    "resample_unit",
    "n_replicates",
    "ci_level",
}


# ---------------------------------------------------------------------------
# (a) point metrics: perfect vs random
# ---------------------------------------------------------------------------
def test_point_metric_row_perfect_ranking():
    n = 100
    correct = np.zeros(n, dtype=np.int64)
    correct[:50] = 1
    confidence = correct.astype(np.float64)  # perfect: confidence orders correctness

    row = point_metric_row(confidence, correct)
    assert row["auroc_correct"] == 1.0
    assert abs(row["e_aurc"]) < 0.02
    assert row["rer_at_50"] == 1.0
    # AURC matches the oracle bound up to finite-sample trapezoid discretisation
    assert abs(row["aurc"] - row["aurc_oracle"]) < 1e-3
    # probability-only metrics are None when no probability is supplied
    assert row["ece_adaptive"] is None
    assert row["brier_binary"] is None
    assert row["nll_binary"] is None


def test_point_metric_row_random_ranking_and_probability():
    rng = np.random.default_rng(0)
    n = 400
    correct = (rng.random(n) < 0.5).astype(np.int64)
    confidence = rng.random(n)

    row = point_metric_row(confidence, correct)
    assert 0.4 <= row["auroc_correct"] <= 0.6
    assert all(row[key] is None for key in evaluate.PROBABILITY_METRICS)

    with_prob = point_metric_row(confidence, correct, probability=confidence)
    for key in evaluate.PROBABILITY_METRICS:
        assert with_prob[key] is not None
        assert np.isfinite(with_prob[key])


# ---------------------------------------------------------------------------
# (b) hand-computed RER
# ---------------------------------------------------------------------------
def test_rer_hand_computed():
    confidence = np.array([0.9, 0.3, 0.2, 0.1])
    correct = np.array([1, 0, 0, 0])
    # base risk = 0.75; the most-confident 50% (ceil(0.5*4)=2) are [1, 0] -> risk 0.5
    # rer = (0.75 - 0.5) / 0.75 = 1/3
    assert rer_at_coverage(confidence, correct, 0.5) == pytest.approx(1.0 / 3.0)
    row = point_metric_row(confidence, correct)
    assert row["rer_at_50"] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# (c) cross-K bootstrap rows: image clustering, A == B, relative worsening
# ---------------------------------------------------------------------------
def test_cross_k_bootstrap_row_image_clustered_and_identical():
    rng = np.random.default_rng(3)
    n = 200
    clusters = rng.integers(0, 40, size=n)
    confidence = rng.random(n)
    correct = (rng.random(n) < 0.5).astype(np.int64)
    n_clusters = int(np.unique(clusters).size)

    rows = cross_k_bootstrap_row(
        confidence,
        correct,
        confidence,
        correct,
        clusters,
        eval_split="testA",
        k_lo=5,
        k_hi=50,
        replicates=50,
        seed=7,
    )

    assert all(r["resample_unit"] == "image" for r in rows)
    assert all(r["n_clusters"] == n_clusters for r in rows)
    assert all(r["eval_split"] == "testA" for r in rows)
    assert all(r["K_lo"] == 5 and r["K_hi"] == 50 for r in rows)

    absolute = [r for r in rows if r["diff_kind"] == "absolute"]
    relative = [r for r in rows if r["diff_kind"] == "relative"]
    assert {r["metric"] for r in absolute} == {"auroc_correct", "e_aurc", "rer_at_50", "rer_at_80"}
    assert len(relative) == 1 and relative[0]["metric"] == "e_aurc"

    for r in absolute:
        assert r["diff"] == 0.0
        assert r["ci_low"] == 0.0 and r["ci_high"] == 0.0
        assert r["mean_lo"] == pytest.approx(r["mean_hi"])
        assert ABS_KEYS.issubset(r.keys())

    rel_row = relative[0]
    assert {"worsening", "worsening_ci_low", "worsening_ci_high"}.issubset(rel_row.keys())
    assert rel_row["worsening"] == pytest.approx(0.0)
    assert rel_row["worsening_ci_low"] <= rel_row["worsening_ci_high"]


# ---------------------------------------------------------------------------
# (d) ratio -> unified worsening mapping (sorted CI endpoints)
# ---------------------------------------------------------------------------
def test_worsening_mapping_and_sorted_ci():
    assert worsening_from_ratio(0.25) == pytest.approx(-0.2)
    assert worsening_from_ratio(-0.5) == pytest.approx(1.0)
    assert worsening_from_ratio(0.0) == pytest.approx(0.0)

    lo, hi = worsening_ci_from_ratio_ci(0.1, 0.5)
    # w is monotonically *decreasing* in t, so the mapped CI is min/max sorted:
    # low <- t=0.5, high <- t=0.1
    assert lo == pytest.approx(worsening_from_ratio(0.5))
    assert hi == pytest.approx(worsening_from_ratio(0.1))
    assert lo <= hi

    # endpoint order must not matter: the interval is always sorted
    assert worsening_ci_from_ratio_ci(0.5, 0.1) == (lo, hi)


# ---------------------------------------------------------------------------
# (e) sufficiency gate: the four canonical outcomes
# ---------------------------------------------------------------------------
def test_sufficiency_verdict_go():
    cells = [
        _cell("testA", 20, e_worsen=0.25, e_lo=0.05, e_hi=0.4, rer_drop=12.0, rer_lo=2.0),
        _cell("testB", 50, e_worsen=0.30, e_lo=0.06, e_hi=0.5, rer_drop=15.0, rer_lo=3.0),
    ]
    out = sufficiency_verdict(
        cells, k50_point={"auroc": 0.72, "e_aurc": 0.20, "rer_at_50": 0.30}
    )
    assert out["verdict"] == "GO_candidate_embeddings"
    assert out["n_failing_ood_cells"] == 2
    assert set(out["route_A_cells"]) == {"testA/K20", "testB/K50"}
    assert out["override"] is False
    assert out["no_go_sufficient"] is False


def test_sufficiency_verdict_no_go_sufficient():
    pooled = _cell(
        "__pooled_test__", 50, auroc_drop=0.01, auroc_lo=-0.01, e_worsen=0.10, e_lo=-0.05, rer_drop=2.0, rer_lo=-1.0
    )
    ood = _cell(
        "testA", 20, auroc_drop=0.01, auroc_lo=-0.01, e_worsen=0.10, e_lo=-0.05, rer_drop=2.0, rer_lo=-1.0
    )
    out = sufficiency_verdict(
        [pooled, ood], k50_point={"auroc": 0.72, "e_aurc": 0.15, "rer_at_50": 0.40}
    )
    assert out["verdict"] == "NO_GO_candidate_embeddings"
    assert out["no_go_sufficient"] is True
    assert out["n_failing_ood_cells"] == 0


def test_sufficiency_verdict_override():
    out = sufficiency_verdict(
        [], k50_point={"auroc": 0.90, "e_aurc": 0.02, "rer_at_50": 0.75}
    )
    assert out["verdict"] == "practically_solved_by_score_only"
    assert out["override"] is True


def test_sufficiency_verdict_inconclusive():
    ood = _cell(
        "testA", 20, auroc_drop=0.01, auroc_lo=-0.01, e_worsen=0.10, e_lo=-0.05, rer_drop=2.0, rer_lo=-1.0
    )
    out = sufficiency_verdict(
        [ood], k50_point={"auroc": 0.72, "e_aurc": 0.15, "rer_at_50": 0.40}
    )
    assert out["verdict"] == "inconclusive"
    assert out["override"] is False
    assert out["no_go_sufficient"] is False


# ---------------------------------------------------------------------------
# (f) model-vs-model rows: schema + identical models
# ---------------------------------------------------------------------------
def test_model_vs_model_bootstrap_row_schema_and_identical():
    rng = np.random.default_rng(5)
    n = 150
    clusters = rng.integers(0, 30, size=n)
    confidence = rng.random(n)
    correct = (rng.random(n) < 0.5).astype(np.int64)

    rows = model_vs_model_bootstrap_row(
        confidence,
        correct,
        confidence,
        correct,
        clusters,
        eval_split="testA",
        K=5,
        model_a="MSP",
        model_b="StatsLogistic",
        replicates=50,
        seed=1,
    )
    assert len(rows) == len(evaluate.PRIMARY_METRICS)
    for row in rows:
        assert MODEL_KEYS.issubset(row.keys())
        assert row["K"] == 5
        assert row["model_a"] == "MSP" and row["model_b"] == "StatsLogistic"
        assert row["resample_unit"] == "image"
        assert row["diff"] == 0.0
        assert row["ci_low"] == 0.0 and row["ci_high"] == 0.0
