"""Tests for the Phase 0A.1 base-rate-aware metrics.

Covers the new selective-prediction helpers (:func:`oracle_aurc`,
:func:`e_aurc`, :func:`rer_at_coverage`) and the discrimination metrics
(:mod:`ccg.metrics.discrimination`).  The discrimination metrics are
cross-checked against scikit-learn on random data with many ties.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from ccg.metrics.discrimination import auprc_correct, auroc_correct
from ccg.metrics.selective import (
    aurc,
    e_aurc,
    oracle_aurc,
    rer_at_coverage,
    risk_coverage_curve,
)


# ---------------------------------------------------------------------------
# oracle_aurc / e_aurc
# ---------------------------------------------------------------------------
def test_oracle_aurc_formula_and_endpoints():
    assert oracle_aurc(0.0) == 0.0
    assert oracle_aurc(1.0) == 1.0
    assert oracle_aurc(0.5) == pytest.approx(0.5 + 0.5 * np.log(0.5))
    assert oracle_aurc(0.25) == pytest.approx(0.25 + 0.75 * np.log(0.75))
    # always below the random baseline, and monotone increasing
    values = [oracle_aurc(r) for r in np.linspace(0.0, 1.0, 21)]
    assert all(v <= r + 1e-12 for v, r in zip(values, np.linspace(0.0, 1.0, 21)))
    assert np.all(np.diff(values) >= -1e-12)


def test_oracle_aurc_validates_input():
    with pytest.raises(ValueError):
        oracle_aurc(-0.1)
    with pytest.raises(ValueError):
        oracle_aurc(1.5)


def _perfect_conf(corr: np.ndarray) -> np.ndarray:
    """Confidence that orders every correct sample above every wrong one."""
    return corr.astype(np.float64)


def test_e_aurc_zero_for_perfect_ordering():
    rng = np.random.default_rng(0)
    corr = np.concatenate([np.ones(700), np.zeros(1300)])
    rng.shuffle(corr)
    conf = _perfect_conf(corr)
    coverage, risk = risk_coverage_curve(conf, corr)
    value = e_aurc(aurc(coverage, risk), float(np.mean(corr)))
    assert value == pytest.approx(0.0, abs=1e-2)


def test_e_aurc_positive_for_random_ordering():
    rng = np.random.default_rng(1)
    n = 3000
    corr = (rng.random(n) < 0.5).astype(np.float64)
    conf = rng.random(n)  # independent of correctness
    coverage, risk = risk_coverage_curve(conf, corr)
    value = e_aurc(aurc(coverage, risk), float(np.mean(corr)))
    assert value > 0.10


# ---------------------------------------------------------------------------
# rer_at_coverage
# ---------------------------------------------------------------------------
def test_rer_one_for_perfect_ordering():
    conf = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    corr = np.array([0.0, 0.0, 1.0, 1.0])
    # accepted subset is error-free at both 25% and 50% coverage
    assert rer_at_coverage(conf, corr, 0.25) == pytest.approx(1.0)
    assert rer_at_coverage(conf, corr, 0.5) == pytest.approx(1.0)


def test_rer_near_zero_for_constant_confidence():
    # 100 interleaved correct/wrong samples with a constant confidence: the
    # ordering degenerates to the input order, so the accepted subset has the
    # same accuracy as the full set and the risk erosion is exactly zero
    corr = np.tile([1.0, 0.0], 50)
    conf = np.full(corr.shape[0], 0.5)
    for level in (0.5, 0.8, 0.9):
        assert rer_at_coverage(conf, corr, level) == pytest.approx(0.0, abs=1e-12)


def test_rer_zero_when_no_base_error():
    corr = np.ones(6)
    conf = np.linspace(0.1, 0.9, 6)
    assert rer_at_coverage(conf, corr, 0.5) == 0.0


# ---------------------------------------------------------------------------
# auroc_correct / auprc_correct
# ---------------------------------------------------------------------------
def test_auroc_perfect_inverted_and_all_tie():
    conf = np.array([0.1, 0.2, 0.8, 0.9])
    corr = np.array([0, 0, 1, 1])
    assert auroc_correct(conf, corr) == pytest.approx(1.0)
    assert auroc_correct(conf, corr[::-1]) == pytest.approx(0.0)
    const = np.full(6, 0.5)
    mixed = np.array([0, 1, 0, 1, 0, 1])
    assert auroc_correct(const, mixed) == pytest.approx(0.5)


def test_auroc_nan_for_empty_or_single_class():
    assert np.isnan(auroc_correct([], []))
    assert np.isnan(auroc_correct([0.1, 0.2, 0.3], [1, 1, 1]))
    assert np.isnan(auroc_correct([0.1, 0.2, 0.3], [0, 0, 0]))


def test_auprc_perfect_and_nan_for_degenerate():
    conf = np.array([0.1, 0.2, 0.8, 0.9])
    corr = np.array([0, 0, 1, 1])
    assert auprc_correct(conf, corr) == pytest.approx(1.0)
    assert np.isnan(auprc_correct([], []))
    assert np.isnan(auprc_correct([0.1, 0.2], [1, 1]))
    assert np.isnan(auprc_correct([0.1, 0.2], [0, 0]))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_matches_sklearn_no_ties(seed):
    rng = np.random.default_rng(seed)
    n = 800
    conf = rng.random(n)
    corr = (rng.random(n) < 0.4).astype(int)
    if corr.min() == corr.max():  # pragma: no cover - practically impossible
        corr[0] = 1 - corr[0]
    assert auroc_correct(conf, corr) == pytest.approx(roc_auc_score(corr, conf), abs=1e-10)
    assert auprc_correct(conf, corr) == pytest.approx(
        average_precision_score(corr, conf), abs=1e-10
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_matches_sklearn_with_heavy_ties(seed):
    rng = np.random.default_rng(seed + 100)
    n = 600
    conf = np.round(rng.random(n), 1)  # 11 distinct values -> many ties
    corr = (rng.random(n) < 0.5).astype(int)
    if corr.min() == corr.max():  # pragma: no cover
        corr[0] = 1 - corr[0]
    assert auroc_correct(conf, corr) == pytest.approx(roc_auc_score(corr, conf), abs=1e-10)
    assert auprc_correct(conf, corr) == pytest.approx(
        average_precision_score(corr, conf), abs=1e-10
    )


def test_all_scores_tied_with_ties_in_labels():
    conf = np.zeros(10)
    corr = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    # 3 positives of 10 -> AUROC 0.5; AUPRC == prevalence (random ranking)
    assert auroc_correct(conf, corr) == pytest.approx(0.5)
    assert auprc_correct(conf, corr) == pytest.approx(0.3)
