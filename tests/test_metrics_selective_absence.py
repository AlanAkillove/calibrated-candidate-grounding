"""Synthetic tests for :mod:`ccg.metrics.selective` and :mod:`ccg.metrics.absence`.

Covers Task 5 scenarios: all-correct / all-wrong risk-coverage boundaries,
constant confidence degeneracy, AUROC / AUPRC extremes, FPR@TPR, NONE
precision/recall/F1 and the false-selection-rate = 0.5 construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from ccg.metrics import absence as absence_mod
from ccg.metrics.absence import (
    auprc,
    auroc,
    false_selection_rate,
    fpr_at_tpr,
    none_f1,
    none_precision,
    none_recall,
    roc_points,
)
from ccg.metrics.calibration import brier_binary, confidence_accuracy_gap, confidence_from_scores
from ccg.metrics.ranking import top1_accuracy
from ccg.metrics.selective import (
    aurc,
    risk_at_coverage,
    risk_coverage_curve,
    selective_accuracy,
    selective_risk,
)

TOL = 1e-12


def _trapz_area(x: np.ndarray, y: np.ndarray) -> float:
    """Plain trapezoidal area, independent of numpy / sklearn versions."""
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0))


# --------------------------------------------------------------------------- #
# risk - coverage / AURC / selective accuracy
# --------------------------------------------------------------------------- #
def test_all_correct_risk_is_zero() -> None:
    conf = np.linspace(0.2, 0.9, 20)
    corr = np.ones(20)
    coverage, risk = risk_coverage_curve(conf, corr)
    assert np.all(risk == 0.0)
    assert aurc(coverage, risk) == pytest.approx(0.0, abs=TOL)
    assert risk_at_coverage(conf, corr, 0.5) == pytest.approx(0.0, abs=TOL)
    assert selective_accuracy(conf, corr, 0.37) == pytest.approx(1.0)


def test_all_wrong_risk_is_one() -> None:
    conf = np.linspace(0.2, 0.9, 20)
    corr = np.zeros(20)
    coverage, risk = risk_coverage_curve(conf, corr)
    assert np.all(risk == 1.0)
    assert aurc(coverage, risk) == pytest.approx(coverage[-1] - coverage[0], abs=TOL)
    assert selective_accuracy(conf, corr, 0.9) == pytest.approx(0.0)
    assert risk_at_coverage(conf, corr, 0.9) == pytest.approx(1.0, abs=TOL)


def test_confidently_sorted_curve_is_monotone() -> None:
    # the 10 most confident samples are exactly the correct ones
    conf = np.array([0.95, 0.94, 0.93, 0.92, 0.91, 0.30, 0.29, 0.28, 0.27, 0.26])
    corr = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
    coverage, risk = risk_coverage_curve(conf, corr)
    assert coverage[0] == pytest.approx(0.1) and coverage[-1] == pytest.approx(1.0)
    assert np.all(risk[:5] == 0.0)
    assert risk[5] == pytest.approx(1 / 6)
    assert risk[-1] == pytest.approx(0.5)
    assert np.all(np.diff(risk[:6]) >= 0)  # perfect confidence ranking -> risk rises steadily
    # AURC is the trapezoid over the measured points
    expected_area = _trapz_area(coverage, risk)
    assert aurc(coverage, risk) == pytest.approx(expected_area, abs=1e-12)
    assert selective_accuracy(conf, corr, 0.5) == pytest.approx(1.0)
    assert selective_accuracy(conf, corr, 0.6) == pytest.approx(5 / 6)
    assert selective_risk(conf, corr, 0.6) == pytest.approx(1 / 6)


def test_constant_confidence_degenerates_gracefully() -> None:
    conf = np.full(100, 0.6)
    corr = np.tile(np.array([1.0, 0.0]), 50)
    coverage, risk = risk_coverage_curve(conf, corr)
    assert coverage.shape == risk.shape == (100,)
    assert risk[-1] == pytest.approx(0.5, abs=TOL)
    assert np.all(risk <= 0.5 + 1e-9) and np.all(risk >= 0.5 - 0.52)
    # every achievable coverage level keeps accuracy at ~0.5
    for level in (0.1, 0.25, 0.5, 0.8, 0.95, 1.0):
        assert selective_accuracy(conf, corr, level) == pytest.approx(0.5, abs=0.03)
    # with constant correctness the curve is *exactly* flat
    flat = np.ones(40)
    _, risk_flat = risk_coverage_curve(conf[:40], flat)
    assert np.all(risk_flat == 0.0)
    conf_even = np.full(10, 0.5)
    corr_even = np.tile(np.array([1.0, 0.0]), 5)
    _, r_even = risk_coverage_curve(conf_even, corr_even)
    assert r_even[1] == pytest.approx(0.5, abs=TOL) and r_even[-1] == pytest.approx(0.5, abs=TOL)


def test_selective_helpers_validate_inputs() -> None:
    with pytest.raises(ValueError, match=r"coverage_level"):
        selective_accuracy(np.array([0.5, 0.6]), np.array([1, 0]), 0.0)
    with pytest.raises(ValueError, match=r"coverage_level"):
        risk_at_coverage(np.array([0.5, 0.6]), np.array([1, 0]), 1.5)
    with pytest.raises(ValueError, match="monotonically"):
        aurc(np.array([0.1, 0.9, 0.4]), np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match="differ in length"):
        risk_coverage_curve(np.array([0.5, 0.4]), np.array([1, 0, 1]))


def test_aurc_of_random_model_is_between_all_correct_and_all_wrong() -> None:
    rng = np.random.default_rng(2)
    conf = rng.uniform(0.0, 1.0, size=500)
    corr = (rng.uniform(size=500) < conf).astype(float)  # informative confidence
    coverage, risk = risk_coverage_curve(conf, corr)
    aurc_good = aurc(coverage, risk)
    coverage_bad, risk_bad = risk_coverage_curve(1.0 - conf, corr)  # anti-calibrated
    assert 0.0 < aurc_good < aurc(coverage_bad, risk_bad) < 1.0


# --------------------------------------------------------------------------- #
# AUROC / AUPRC / FPR@TPR
# --------------------------------------------------------------------------- #
PERFECT = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.3, 0.4])  # 4 pos then 4 neg
LABELS = np.array([1, 1, 1, 1, 0, 0, 0, 0])


def test_auroc_extremes() -> None:
    assert auroc(PERFECT, LABELS) == pytest.approx(1.0, abs=TOL)
    assert auroc(-PERFECT, LABELS) == pytest.approx(0.0, abs=TOL)
    # a fully tied score gives the coin-flip value
    assert auroc(np.ones(8), LABELS) == pytest.approx(0.5, abs=TOL)
    rng = np.random.default_rng(4)
    scores = rng.uniform(size=2000)
    labels = (rng.uniform(size=2000) < 0.5).astype(float)
    assert 0.45 <= auroc(scores, labels) <= 0.55


def test_auroc_handles_ties_exactly() -> None:
    scores = np.array([1.0, 1.0, 1.0, 0.0])
    labels = np.array([1, 0, 0, 0])
    # pairs: (tie -> 0.5), (tie -> 0.5), (win -> 1.0) => 2/3
    assert auroc(scores, labels) == pytest.approx(2.0 / 3.0, abs=TOL)


def test_auprc_perfect_and_chance() -> None:
    assert auprc(PERFECT, LABELS) == pytest.approx(1.0, abs=TOL)
    inverted = 0.25 * (1 / 5 + 2 / 6 + 3 / 7 + 4 / 8)
    assert auprc(-PERFECT, LABELS) == pytest.approx(inverted, abs=TOL)
    # positives first -> every incremental recall is earned at precision 1
    scores = np.array([0.9, 0.8, 0.5, 0.4, 0.1])
    labels = np.array([1, 1, 0, 0, 0])
    assert auprc(scores, labels) == pytest.approx(1.0, abs=TOL)
    # one bad negative interleaved: recall +0.5 at precision 1, then +0.5 at precision 2/3
    scores2 = np.array([0.9, 0.8, 0.7, 0.6])
    labels2 = np.array([1, 0, 1, 0])
    assert auprc(scores2, labels2) == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3), abs=TOL)


def test_fpr_at_tpr_known_operating_points() -> None:
    assert fpr_at_tpr(PERFECT, LABELS, tpr=0.95) == pytest.approx(0.0, abs=TOL)
    assert fpr_at_tpr(-PERFECT, LABELS, tpr=0.95) == pytest.approx(1.0, abs=TOL)
    # one negative ranked above all positives -> to catch 4/4 positives we eat 1/4 FPR
    scores = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.1, 0.05, 0.02])
    labels = np.array([0, 1, 1, 1, 1, 0, 0, 0])
    assert fpr_at_tpr(scores, labels, tpr=0.95) == pytest.approx(0.25, abs=TOL)
    assert fpr_at_tpr(PERFECT, LABELS, tpr=0.5) == pytest.approx(0.0, abs=TOL)
    with pytest.raises(ValueError, match="tpr"):
        fpr_at_tpr(PERFECT, LABELS, tpr=0.0)


def test_roc_points_start_and_end_correctly() -> None:
    fpr, tpr, thr = roc_points(PERFECT, LABELS)
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == 1.0 and tpr[-1] == 1.0
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
    assert thr.shape == fpr.shape


def test_absence_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="at least one positive"):
        auroc(np.array([0.1, 0.2]), np.array([1, 1]))
    with pytest.raises(ValueError, match="binary"):
        auprc(np.array([0.1, 0.2]), np.array([0.5, 1.0]))
    with pytest.raises(ValueError, match="differ in length"):
        auroc(np.array([0.1, 0.2, 0.3]), np.array([1, 0]))


@pytest.mark.skipif(not absence_mod.HAS_SKLEARN, reason="scikit-learn not installed")
def test_numpy_fallback_matches_sklearn() -> None:  # pragma: no cover - env dependent
    rng = np.random.default_rng(9)
    scores = rng.normal(size=400)
    labels = (rng.uniform(size=400) < 0.35).astype(float)
    labels[0] = 1.0
    scores[-1] = -4.0
    assert absence_mod._auroc_numpy(scores, labels) == pytest.approx(
        absence_mod.roc_auc_score(labels, scores), abs=1e-12
    )
    assert absence_mod._auprc_numpy(scores, labels) == pytest.approx(
        absence_mod.average_precision_score(labels, scores), abs=1e-12
    )
    fpr_np, tpr_np, _ = absence_mod._roc_points(labels, scores)
    fpr_sk, tpr_sk, _ = absence_mod.roc_curve(labels, scores)
    assert fpr_np[0] == fpr_sk[0] == 0.0 and tpr_np[-1] == tpr_sk[-1] == 1.0
    # scikit-learn drops the TP-only steps of the staircase, so the point sets differ
    # while the curve (and therefore the area) is identical
    area_np = _trapz_area(fpr_np, tpr_np)
    area_sk = _trapz_area(fpr_sk, tpr_sk)
    assert area_np == pytest.approx(area_sk, abs=1e-12)
    assert area_np == pytest.approx(auroc(scores, labels), abs=1e-12)
    # identical operating point for the FPR@TPR family
    for level in (0.5, 0.8, 0.95, 1.0):
        assert fpr_at_tpr(scores, labels, tpr=level) == pytest.approx(
            float(fpr_sk[np.argmax(tpr_sk >= level)]), abs=TOL
        )


# --------------------------------------------------------------------------- #
# NONE decisions and false selection rate
# --------------------------------------------------------------------------- #
def test_none_precision_recall_f1() -> None:
    pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    true = np.array([1, 0, 0, 1, 1, 0, 1, 0])
    # predicted NONE: 3 of which 2 are true NONE -> precision 2/3
    # true NONE: 4 of which 2 predicted -> recall 1/2
    assert none_precision(pred, true) == pytest.approx(2 / 3, abs=TOL)
    assert none_recall(pred, true) == pytest.approx(0.5, abs=TOL)
    assert none_f1(pred, true) == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5), abs=TOL)
    # perfect detector
    assert none_f1(true, true) == pytest.approx(1.0, abs=TOL)
    # zero-division conventions
    assert none_precision(np.zeros(4, dtype=bool), true[:4]) == pytest.approx(0.0, abs=TOL)
    assert none_recall(np.ones(8, dtype=bool), np.zeros(8, dtype=bool)) == pytest.approx(0.0)
    assert none_f1(np.zeros(8, dtype=bool), true) == pytest.approx(0.0)


def test_false_selection_rate_half() -> None:
    # 4 target-absent samples: 2 abstain (pred_index == -1), 2 select a candidate -> 0.5
    pred = np.array([-1, 0, -1, 2])
    target = np.array([-1, -1, -1, -1])
    present = np.array([False, False, False, False])
    assert false_selection_rate(pred, target, present) == pytest.approx(0.5, abs=TOL)
    # target-present rows are excluded from the denominator entirely
    pred2 = np.array([3, 3, -1, 0, -1, 2])
    target2 = np.array([3, 3, -1, -1, -1, -1])
    present2 = np.array([True, True, False, False, False, False])
    assert false_selection_rate(pred2, target2, present2) == pytest.approx(0.5, abs=TOL)
    # never abstaining is the worst case, always abstaining the best
    assert false_selection_rate(np.full(4, -1), target, present) == 0.0
    assert false_selection_rate(np.zeros(4, dtype=int), target, present) == 1.0
    with pytest.raises(ValueError, match="at least one target-absent"):
        false_selection_rate(np.array([0, 1]), np.array([0, 1]), np.array([True, True]))
    with pytest.raises(ValueError, match="target_present"):
        false_selection_rate(np.array([0]), np.array([-1]), np.array([True]))


def test_metrics_compose_on_a_synthetic_candidate_set() -> None:
    """End-to-end smoke test: scores -> confidence -> calibration + selection."""
    rng = np.random.default_rng(13)
    n, k = 300, 10
    scores = rng.normal(0.0, 1.0, size=(n, k))
    target = rng.integers(0, k, size=n)
    scores[np.arange(n), target] += 1.2  # a weak but real signal
    conf, pred = confidence_from_scores(scores, temperature=1.5)
    corr = (pred == target).astype(float)
    assert conf.shape == (n,) and np.all(conf >= 1 / k - 1e-9) and np.all(conf <= 1.0)
    accuracy = float(np.mean(corr))
    assert selective_accuracy(conf, corr, 1.0) == pytest.approx(accuracy, abs=TOL)
    coverage, risk = risk_coverage_curve(conf, corr)
    assert risk[-1] == pytest.approx(1 - accuracy, abs=TOL)
    assert risk_at_coverage(conf, corr, 0.9) <= 1.0
    assert 0.0 <= aurc(coverage, risk) <= 1.0
    # calibration numbers are consistent with each other
    assert confidence_accuracy_gap(conf, corr) == pytest.approx(float(np.mean(conf)) - accuracy)
    assert brier_binary(conf, corr) >= 0.0
    # the softmax-argmax route and the ranking route agree on Top-1
    assert top1_accuracy(scores, target) == pytest.approx(accuracy, abs=1e-12)
