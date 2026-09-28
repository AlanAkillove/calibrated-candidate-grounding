"""Unit tests for :mod:`ccg.semantic.evaluate` - pure NumPy, no real caches.

Covers the A7 ratio -> reduction mapping, the frozen A7.6 semantic gate
(table-driven STRONG / PASS / NO_GO / INCONCLUSIVE plus None-robustness and
threshold overrides) and the model-vs-model E-AURC ratio bootstrap row
(identical scorers, schema / field discipline and K-as-metadata).
"""

from __future__ import annotations

import csv
import io

import numpy as np
import pytest

from ccg.reliability import evaluate as reval
from ccg.semantic import evaluate
from ccg.semantic.evaluate import (
    RATIO_EXTRA_FIELDS,
    SEMANTIC_THRESHOLDS,
    model_vs_model_e_aurc_ratio_row,
    reduction_ci_from_ratio_ci,
    reduction_from_ratio,
    semantic_gate,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cell(
    *,
    delta_auroc=None,
    delta_auroc_ci_low=None,
    e_aurc_reduction=None,
    e_aurc_reduction_ci_low=None,
    rer50_gain_pp=None,
    rer50_gain_pp_ci_low=None,
):
    return {
        "delta_auroc": delta_auroc,
        "delta_auroc_ci_low": delta_auroc_ci_low,
        "e_aurc_reduction": e_aurc_reduction,
        "e_aurc_reduction_ci_low": e_aurc_reduction_ci_low,
        "rer50_gain_pp": rer50_gain_pp,
        "rer50_gain_pp_ci_low": rer50_gain_pp_ci_low,
    }


def _clustered_data(seed=0, n=60, n_clusters=6):
    rng = np.random.default_rng(seed)
    clusters = np.repeat(np.arange(n_clusters), n // n_clusters)
    confidence = rng.random(n)
    correct = (rng.random(n) < 0.5).astype(np.int64)
    return confidence, correct, clusters


# ---------------------------------------------------------------------------
# (1) ratio -> reduction mapping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("a,b", [(0.3, 0.2), (0.2, 0.3), (0.5, 0.4), (0.11, 0.10)])
def test_reduction_from_ratio_identity(a, b):
    t = (a - b) / b
    assert reduction_from_ratio(t) == pytest.approx((b - a) / b)
    # reduction is exactly the negation of the ratio
    assert reduction_from_ratio(t) == pytest.approx(-t)


def test_reduction_ci_from_ratio_ci_endpoint_mapping_and_sorted():
    # monotone decreasing map: t=0.1 -> -0.1, t=0.5 -> -0.5, then sorted
    low, high = reduction_ci_from_ratio_ci(0.1, 0.5)
    assert low == pytest.approx(-0.5)
    assert high == pytest.approx(-0.1)
    assert low <= high

    # endpoint order must not matter
    assert reduction_ci_from_ratio_ci(0.5, 0.1) == (low, high)


def test_thresholds_frozen_values():
    assert SEMANTIC_THRESHOLDS["auroc_gain"] == 0.02
    assert SEMANTIC_THRESHOLDS["auroc_gain_strong"] == 0.03
    assert SEMANTIC_THRESHOLDS["e_aurc_reduction"] == 0.10
    assert SEMANTIC_THRESHOLDS["e_aurc_reduction_strong"] == 0.20
    assert SEMANTIC_THRESHOLDS["rer50_gain_pp"] == 5.0
    assert SEMANTIC_THRESHOLDS["rer50_gain_pp_strong"] == 10.0
    assert SEMANTIC_THRESHOLDS["seed_sig_min"] == 2
    assert SEMANTIC_THRESHOLDS["seed_total"] == 3
    assert RATIO_EXTRA_FIELDS == ("reduction", "reduction_ci_low", "reduction_ci_high")


# ---------------------------------------------------------------------------
# (2) semantic gate: the four canonical outcomes
# ---------------------------------------------------------------------------
def test_semantic_gate_strong():
    cells = {
        20: _cell(delta_auroc=0.025, delta_auroc_ci_low=0.006, e_aurc_reduction=0.12,
                  e_aurc_reduction_ci_low=0.01, rer50_gain_pp=6.0, rer50_gain_pp_ci_low=0.5),
        50: _cell(delta_auroc=0.035, delta_auroc_ci_low=0.008, e_aurc_reduction=0.25,
                  e_aurc_reduction_ci_low=0.02, rer50_gain_pp=12.0, rer50_gain_pp_ci_low=1.0),
    }
    out = semantic_gate(cells, [0.01, 0.02, -0.005], seed_mean_delta=0.02)
    assert out["verdict"] == "STRONG"
    assert out["strong"] is True
    assert out["pass"] is True
    assert out["no_go"] is False
    assert out["cell_ok"] == {20: True, 50: True}
    assert out["n_seed_significant"] == 2
    assert out["seed_sig_required"] == 2
    assert out["seed_mean_delta"] == pytest.approx(0.02)
    assert out["cell_conditions"][50]["cell_ok"] is True


def test_semantic_gate_pass():
    cells = {
        20: _cell(delta_auroc=0.025, delta_auroc_ci_low=0.004, e_aurc_reduction=0.12,
                  e_aurc_reduction_ci_low=0.01, rer50_gain_pp=0.0, rer50_gain_pp_ci_low=0.0),
        50: _cell(delta_auroc=0.030, delta_auroc_ci_low=0.006, e_aurc_reduction=0.12,
                  e_aurc_reduction_ci_low=0.01, rer50_gain_pp=0.0, rer50_gain_pp_ci_low=0.0),
    }
    out = semantic_gate(cells, [0.01, 0.02, -0.005], seed_mean_delta=0.01)
    assert out["verdict"] == "PASS"
    assert out["strong"] is False and out["pass"] is True and out["no_go"] is False


def test_semantic_gate_no_go():
    cells = {
        20: _cell(delta_auroc=0.005, delta_auroc_ci_low=-0.002, e_aurc_reduction=0.02,
                  e_aurc_reduction_ci_low=0.0, rer50_gain_pp=1.5, rer50_gain_pp_ci_low=0.0),
        50: _cell(delta_auroc=0.005, delta_auroc_ci_low=-0.002, e_aurc_reduction=0.02,
                  e_aurc_reduction_ci_low=0.0, rer50_gain_pp=1.5, rer50_gain_pp_ci_low=0.0),
    }
    out = semantic_gate(cells, [0.0, -0.01, 0.0], seed_mean_delta=0.0)
    assert out["verdict"] == "NO_GO"
    assert out["no_go"] is True and out["pass"] is False and out["strong"] is False
    assert out["cell_ok"] == {20: False, 50: False}


def test_semantic_gate_inconclusive_gray_zone():
    cells = {
        20: _cell(delta_auroc=0.015, delta_auroc_ci_low=0.002, e_aurc_reduction=0.08,
                  e_aurc_reduction_ci_low=0.005, rer50_gain_pp=4.0, rer50_gain_pp_ci_low=0.4),
        50: _cell(delta_auroc=0.015, delta_auroc_ci_low=0.002, e_aurc_reduction=0.08,
                  e_aurc_reduction_ci_low=0.005, rer50_gain_pp=4.0, rer50_gain_pp_ci_low=0.4),
    }
    out = semantic_gate(cells, [], seed_mean_delta=0.0)
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["strong"] is False and out["pass"] is False and out["no_go"] is False
    assert out["cell_ok"] == {20: False, 50: False}


# ---------------------------------------------------------------------------
# (2b) None / missing robustness and seed counting
# ---------------------------------------------------------------------------
def test_semantic_gate_none_and_missing_never_raises():
    out = semantic_gate(None, None, seed_mean_delta=None)
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["cell_ok"] == {20: None, 50: None}
    assert out["n_seed_significant"] == 0
    assert out["seed_mean_delta"] is None

    out = semantic_gate({20: {"delta_auroc": None}, 50: None}, [], seed_mean_delta="x")
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["cell_ok"] == {20: False, 50: None}
    assert out["seed_mean_delta"] is None

    # a present cell with NaN fields is "not satisfied", not an exception
    nan = float("nan")
    out = semantic_gate(
        {20: _cell(delta_auroc=nan, delta_auroc_ci_low=nan), 50: _cell(delta_auroc=nan)},
        [nan, None],
        seed_mean_delta=nan,
    )
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["cell_ok"] == {20: False, 50: False}
    assert out["n_seed_significant"] == 0


def test_semantic_gate_seed_significant_count():
    seeds = [0.01, None, 0.0, 0.02, float("nan"), 0.03, -0.001]
    out = semantic_gate({}, seeds, seed_mean_delta=0.0)
    assert out["n_seed_significant"] == 3
    assert out["seed_sig_required"] == 2


def test_semantic_gate_thresholds_override_effective():
    cells = {
        20: _cell(delta_auroc=0.025, delta_auroc_ci_low=0.004, e_aurc_reduction=0.12,
                  e_aurc_reduction_ci_low=0.01, rer50_gain_pp=6.0, rer50_gain_pp_ci_low=0.5),
        50: _cell(delta_auroc=0.025, delta_auroc_ci_low=0.004, e_aurc_reduction=0.12,
                  e_aurc_reduction_ci_low=0.01, rer50_gain_pp=6.0, rer50_gain_pp_ci_low=0.5),
    }
    # default: both cells pass -> PASS
    default = semantic_gate(cells, [0.01, 0.02, -0.005], seed_mean_delta=0.01)
    assert default["verdict"] == "PASS"

    # raising the auroc bar rejects both cells -> INCONCLUSIVE
    strict = semantic_gate(
        cells, [0.01, 0.02, -0.005], seed_mean_delta=0.01, thresholds={"auroc_gain": 0.05}
    )
    assert strict["verdict"] == "INCONCLUSIVE"
    assert strict["thresholds"]["auroc_gain"] == 0.05

    # raising the seed bar rejects the 2/3 seed evidence -> not PASS
    seed_strict = semantic_gate(
        cells, [0.01, 0.02, -0.005], seed_mean_delta=0.01, thresholds={"seed_sig_min": 3}
    )
    assert seed_strict["verdict"] == "INCONCLUSIVE"
    assert seed_strict["seed_sig_required"] == 3


# ---------------------------------------------------------------------------
# (3) model-vs-model E-AURC ratio row
# ---------------------------------------------------------------------------
def test_ratio_row_identical_scorers_and_schema():
    confidence, correct, clusters = _clustered_data(seed=0)
    row = model_vs_model_e_aurc_ratio_row(
        confidence, correct, confidence, correct, clusters,
        eval_split="__pooled_test__", K=50, model_sem="E3-full",
        model_score="StatsLogistic", replicates=200, seed=3,
    )
    assert row["metric"] == "e_aurc"
    assert row["diff_kind"] == "relative"
    assert row["model_a"] == "E3-full" and row["model_b"] == "StatsLogistic"
    assert row["eval_split"] == "__pooled_test__" and row["K"] == 50
    # identical scorers: point estimates are exactly zero
    assert row["diff"] == 0.0
    assert row["reduction"] == 0.0
    assert row["ci_low"] == 0.0 and row["ci_high"] == 0.0
    assert row["reduction_ci_low"] == 0.0 and row["reduction_ci_high"] == 0.0
    assert row["n"] == 60 and row["n_clusters"] == 6
    assert row["resample_unit"] == "image"
    assert row["n_replicates"] == 200 and row["replicates"] == 200
    assert row["seed"] == 3 and row["ci"] == pytest.approx(0.95)
    assert row["ci_level"] == pytest.approx(0.95)

    for key in ("diff", "ci_low", "ci_high", "mean_a", "mean_b",
                "reduction", "reduction_ci_low", "reduction_ci_high"):
        assert np.isfinite(row[key])

    # the full field set is exactly the declared schema (no stray keys)
    assert set(row) == set(evaluate.SEMANTIC_RATIO_FIELDS)
    assert set(evaluate.SEMANTIC_RATIO_FIELDS) == set(row)
    assert "reduction" in RATIO_EXTRA_FIELDS


def test_ratio_row_sign_and_ci_mapping():
    # semantic = perfect ordering, score-only = reversed ordering => reduction > 0
    n = 80
    clusters = np.repeat(np.arange(8), 10)
    correct = np.array([1, 0] * (n // 2), dtype=np.int64)
    sem_conf = correct.astype(np.float64)
    score_conf = 1.0 - correct.astype(np.float64)

    row = model_vs_model_e_aurc_ratio_row(
        sem_conf, correct, score_conf, correct, clusters,
        eval_split="__pooled_test__", K=50, model_sem="E3-full",
        model_score="MSP", replicates=200, seed=9,
    )
    assert row["reduction"] == pytest.approx(-row["diff"])
    assert row["reduction"] > 0.0
    # reduction CI endpoints are the negated, sorted ratio CI endpoints
    assert row["reduction_ci_low"] == pytest.approx(-row["ci_high"])
    assert row["reduction_ci_high"] == pytest.approx(-row["ci_low"])
    assert row["reduction_ci_low"] <= row["reduction_ci_high"]


def test_ratio_row_writable_with_reval_fields():
    confidence, correct, clusters = _clustered_data(seed=1)

    # "reval bootstrap field set" derived from the reval model-vs-model row
    reval_row = reval.model_vs_model_bootstrap_row(
        confidence, correct, confidence, correct, clusters,
        eval_split="__pooled_test__", K=50, model_a="a", model_b="b",
        metrics=("e_aurc",), replicates=20, seed=0,
    )[0]
    reval_fields = set(reval_row)
    # our module contributes diff_kind + the echoed provenance + RATIO extras
    fieldnames = sorted(reval_fields | {"diff_kind", "seed", "replicates", "ci"}
                        | set(RATIO_EXTRA_FIELDS))

    row = model_vs_model_e_aurc_ratio_row(
        confidence, correct, confidence, correct, clusters,
        eval_split="__pooled_test__", K=50, model_sem="E3-full",
        model_score="StatsLogistic", replicates=200, seed=3,
    )

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    writer.writerow(row)  # must not raise: no key outside the declared columns
    assert set(fieldnames) == set(row)


def test_ratio_row_K_is_metadata_only():
    confidence, correct, clusters = _clustered_data(seed=2)
    row20 = model_vs_model_e_aurc_ratio_row(
        confidence, correct, confidence, correct, clusters,
        eval_split="testA", K=20, model_sem="E3-full", model_score="MSP",
        replicates=150, seed=5,
    )
    row50 = model_vs_model_e_aurc_ratio_row(
        confidence, correct, confidence, correct, clusters,
        eval_split="testA", K=50, model_sem="E3-full", model_score="MSP",
        replicates=150, seed=5,
    )
    assert row20["K"] == 20 and row50["K"] == 50
    for key in ("diff", "ci_low", "ci_high", "reduction", "reduction_ci_low",
                "reduction_ci_high", "mean_a", "mean_b", "n", "n_clusters"):
        assert row20[key] == row50[key]
