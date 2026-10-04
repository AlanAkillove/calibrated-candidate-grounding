"""Synthetic tests for :mod:`ccg.metrics.calibration` (Task 5 calibration scenarios).

Covers: perfect calibration, overconfidence, underconfidence, all-correct /
all-wrong boundary behaviour, and the adaptive vs equal-width ECE difference.
"""

from __future__ import annotations

import numpy as np
import pytest

from ccg.metrics.calibration import (
    brier_binary,
    confidence_accuracy_gap,
    confidence_from_scores,
    multiclass_brier,
    multiclass_nll,
    nll_binary,
    reliability_table,
    softmax_scores,
    top_label_ece,
    top_label_ece_adaptive,
)
from ccg.metrics.selective import aurc, e_aurc, oracle_aurc, risk_coverage_curve

TOL = 1e-12


def _perfectly_calibrated(n_bins: int = 15, per_bin: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Data with ``P(correct | p) == p`` exactly, bin by bin.

    Bin ``b`` gets ``per_bin`` samples whose confidence is the bin centre
    ``(2b+1)/(2*n_bins)`` and exactly ``2b+1`` of them are correct, so both the
    equal-width and the equal-mass binning see zero confidence/accuracy gap.
    """
    confs, corr = [], []
    for b in range(n_bins):
        p = (2 * b + 1) / (2 * n_bins)
        confs.append(np.full(per_bin, p))
        correct = np.zeros(per_bin)
        correct[: int(round(p * per_bin))] = 1.0
        corr.append(correct)
    return np.concatenate(confs), np.concatenate(corr)


# --------------------------------------------------------------------------- #
# softmax / confidence
# --------------------------------------------------------------------------- #
def test_confidence_from_scores_manual_values() -> None:
    scores = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    conf, pred = confidence_from_scores(scores)
    assert conf.shape == (2,) and pred.shape == (2,)
    assert pred.tolist() == [0, 0]
    e = np.exp(np.array([2.0, 1.0, 0.0]) - 2.0)
    assert conf[0] == pytest.approx(e[0] / e.sum())
    assert conf[1] == pytest.approx(1.0 / 3.0)  # uniform scores
    assert np.allclose(softmax_scores(scores).sum(axis=1), 1.0)


def test_temperature_controls_confidence_only() -> None:
    scores = np.array([[4.0, 1.0, -2.0]])
    conf_cold, pred_cold = confidence_from_scores(scores, temperature=1.0)
    conf_hot, pred_hot = confidence_from_scores(scores, temperature=20.0)
    assert pred_cold.tolist() == pred_hot.tolist() == [0]  # scaling preserves this sample's candidate argmax
    assert conf_hot[0] < conf_cold[0]
    assert conf_cold[0] < 1.0 and conf_hot[0] > 1.0 / 3.0
    conf_sharp, _ = confidence_from_scores(scores, temperature=0.05)
    assert conf_sharp[0] == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="temperature"):
        confidence_from_scores(scores, temperature=0.0)
    with pytest.raises(ValueError, match="temperature"):
        confidence_from_scores(scores, temperature=-1.0)


# --------------------------------------------------------------------------- #
# ECE: perfect calibration / over- / under-confidence
# --------------------------------------------------------------------------- #
def test_perfect_calibration_gives_zero_ece() -> None:
    conf, corr = _perfectly_calibrated()
    assert top_label_ece(conf, corr, n_bins=15) == pytest.approx(0.0, abs=TOL)
    assert top_label_ece_adaptive(conf, corr, n_bins=15) == pytest.approx(0.0, abs=TOL)
    assert confidence_accuracy_gap(conf, corr) == pytest.approx(0.0, abs=TOL)


def test_overconfidence_is_positive_gap_and_ece() -> None:
    conf = np.full(200, 0.9)
    corr = np.zeros(200)
    corr[:100] = 1.0
    assert np.mean(corr) == pytest.approx(0.5)
    assert top_label_ece(conf, corr) == pytest.approx(0.4, abs=TOL)
    # all confidences are tied -> the adaptive binning collapses to a single bin,
    # which is exactly what makes it independent of the input order
    assert top_label_ece_adaptive(conf, corr) == pytest.approx(0.4, abs=TOL)
    assert confidence_accuracy_gap(conf, corr) == pytest.approx(0.4, abs=TOL)
    assert brier_binary(conf, corr) == pytest.approx(0.5 * 0.01 + 0.5 * 0.81, abs=TOL)


def test_adaptive_ece_is_invariant_to_input_order() -> None:
    conf, corr = _sparse_bin_data()
    rng = np.random.default_rng(11)
    for _ in range(5):
        perm = rng.permutation(conf.shape[0])
        assert top_label_ece_adaptive(conf[perm], corr[perm], n_bins=4) == pytest.approx(
            top_label_ece_adaptive(conf, corr, n_bins=4), abs=TOL
        )
        assert top_label_ece(conf[perm], corr[perm], n_bins=2) == pytest.approx(
            top_label_ece(conf, corr, n_bins=2), abs=TOL
        )


def test_underconfidence_has_opposite_sign() -> None:
    conf = np.full(200, 0.3)
    corr = np.zeros(200)
    corr[:100] = 1.0
    assert confidence_accuracy_gap(conf, corr) == pytest.approx(-0.2, abs=TOL)
    assert top_label_ece(conf, corr) == pytest.approx(0.2, abs=TOL)
    assert top_label_ece(conf, corr) * confidence_accuracy_gap(conf, corr) < 0


def test_finite_sample_eaurc_can_be_negative_for_oracle_ordering() -> None:
    # The confidence order is already oracle-perfect. The finite curve uses a
    # trapezoidal area over its two observed points; the reference is continuous.
    confidence = np.asarray([0.8, 0.3])
    correctness = np.asarray([1, 0])
    coverage, risk = risk_coverage_curve(confidence, correctness)

    raw_aurc = aurc(coverage, risk)
    oracle = oracle_aurc(1.0 - float(np.mean(correctness)))
    excess = e_aurc(raw_aurc, float(np.mean(correctness)))

    assert raw_aurc == pytest.approx(0.125, abs=TOL)
    assert oracle == pytest.approx(0.15342640972002736, abs=TOL)
    assert excess == pytest.approx(-0.028426409720027357, abs=TOL)


# --------------------------------------------------------------------------- #
# all-correct / all-wrong boundary behaviour
# --------------------------------------------------------------------------- #
def test_all_correct_boundaries() -> None:
    conf = np.full(50, 0.99)
    corr = np.ones(50)
    assert brier_binary(conf, corr) == pytest.approx(0.0001, abs=TOL)
    assert nll_binary(conf, corr) == pytest.approx(-np.log(0.99), abs=TOL)
    assert top_label_ece(conf, corr) == pytest.approx(0.01, abs=TOL)
    assert confidence_accuracy_gap(conf, corr) == pytest.approx(-0.01, abs=TOL)


def test_all_wrong_stays_finite() -> None:
    conf = np.full(50, 0.99)
    corr = np.zeros(50)
    assert brier_binary(conf, corr) == pytest.approx(0.9801, abs=TOL)
    assert np.isfinite(nll_binary(conf, corr))
    assert nll_binary(conf, corr) == pytest.approx(-np.log(0.01), abs=TOL)
    # perfectly confident and perfectly wrong -> clipped, still finite
    assert np.isfinite(nll_binary(np.ones(5), np.zeros(5)))
    assert nll_binary(np.ones(5), np.zeros(5)) < 30.0
    # perfectly confident and perfectly right -> NLL ~ 0
    assert nll_binary(np.ones(5), np.ones(5)) == pytest.approx(0.0, abs=1e-11)


def test_nll_eps_is_configurable() -> None:
    conf, corr = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    strict = nll_binary(conf, corr, eps=1e-3)
    loose = nll_binary(conf, corr, eps=0.25)
    assert strict > loose
    assert strict == pytest.approx(-np.log(1e-3), abs=TOL)
    with pytest.raises(ValueError, match="eps"):
        nll_binary(conf, corr, eps=0.0)


# --------------------------------------------------------------------------- #
# adaptive vs equal-width binning
# --------------------------------------------------------------------------- #
def _sparse_bin_data() -> tuple[np.ndarray, np.ndarray]:
    """8 samples: 6 @ conf 0.25 (1 correct), 2 @ conf 0.75 (both correct)."""
    conf = np.array([0.25] * 6 + [0.75] * 2)
    corr = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    return conf, corr


def _dense_sparse_data() -> tuple[np.ndarray, np.ndarray]:
    """10 tied-free samples: 8 distinct low confidences (all correct), 2 high (wrong)."""
    conf = np.array([0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.95, 0.96])
    corr = np.array([1.0] * 8 + [0.0, 0.0])
    return conf, corr


def _adaptive_ece_reference(conf: np.ndarray, corr: np.ndarray, n_bins: int) -> float:
    """Naive loop reference for equal-mass binning on tie-free confidence values."""
    order = np.argsort(conf, kind="stable")
    conf_s, corr_s = conf[order], corr[order]
    n = conf.shape[0]
    edges = [int(i * n / n_bins) for i in range(n_bins + 1)]
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        total += (hi - lo) / n * abs(corr_s[lo:hi].mean() - conf_s[lo:hi].mean())
    return float(total)


def test_adaptive_and_equal_width_differ_and_are_correct() -> None:
    conf, corr = _sparse_bin_data()
    # equal width, 2 bins: [0, .5) -> 6 samples (acc 1/6, conf .25); [.5, 1] -> 2 (acc 1, conf .75)
    expected_width = 6 / 8 * abs(1 / 6 - 0.25) + 2 / 8 * abs(1.0 - 0.75)
    assert top_label_ece(conf, corr, n_bins=2) == pytest.approx(expected_width, abs=TOL)
    # equal mass on 6 tied + 2 tied values cannot split the ties -> 2 bins, same answer here
    assert top_label_ece_adaptive(conf, corr, n_bins=4) == pytest.approx(expected_width, abs=TOL)
    assert top_label_ece_adaptive(conf, corr, n_bins=2) == pytest.approx(expected_width, abs=TOL)

    # tie-free data concentrated in two clusters: the two binnings disagree strongly
    conf, corr = _dense_sparse_data()
    expected_ew = 0.8 * abs(1.0 - 0.17) + 0.2 * abs(0.0 - 0.955)
    expected_ad = 0.5 * abs(1.0 - 0.14) + 0.5 * abs(0.6 - 0.514)
    assert top_label_ece(conf, corr, n_bins=2) == pytest.approx(expected_ew, abs=TOL)
    assert top_label_ece_adaptive(conf, corr, n_bins=2) == pytest.approx(expected_ad, abs=TOL)
    assert expected_ew != pytest.approx(expected_ad, abs=1e-6)

    # ... and the implementation matches an independent naive reference on random data
    rng = np.random.default_rng(5)
    conf_r = rng.uniform(0.0, 1.0, size=1000)
    corr_r = (rng.uniform(size=1000) < conf_r).astype(float)
    for nb in (3, 5, 10, 15):
        assert top_label_ece_adaptive(conf_r, corr_r, n_bins=nb) == pytest.approx(
            _adaptive_ece_reference(conf_r, corr_r, nb), abs=TOL
        )


def test_adaptive_bins_are_equal_mass() -> None:
    conf, corr = _dense_sparse_data()
    table = reliability_table(conf, corr, n_bins=2, adaptive=True)
    assert table["counts"].tolist() == [5, 5]
    assert table["n_bins"] == 2
    assert table["bin_edges"][0] == 0.0 and table["bin_edges"][-1] == 1.0
    assert table["bin_edges"][1] == pytest.approx(0.19, abs=TOL)  # midpoint of 0.18 / 0.20
    assert table["avg_confidence"].tolist() == [pytest.approx(0.14), pytest.approx(0.514)]
    assert table["accuracy"].tolist() == [pytest.approx(1.0), pytest.approx(0.6)]
    assert table["ece"] == pytest.approx(top_label_ece_adaptive(conf, corr, n_bins=2), abs=TOL)
    # tied data: bin 4 collapses to 2 non-degenerate bins, each still non-empty
    tied = reliability_table(*_sparse_bin_data(), n_bins=4, adaptive=True)
    assert tied["counts"].tolist() == [6, 2]
    assert tied["n_bins"] == 2
    wide = reliability_table(*_sparse_bin_data(), n_bins=2, adaptive=False)
    assert wide["counts"].tolist() == [6, 2]
    assert wide["bin_edges"].tolist() == [0.0, 0.5, 1.0]
    assert np.all(np.diff(wide["bin_edges"]) > 0)


def test_reliability_table_fields_and_empty_bins() -> None:
    conf, corr = _perfectly_calibrated()
    table = reliability_table(conf, corr, n_bins=15)
    assert set(table) >= {"n_bins", "adaptive", "bin_edges", "counts", "avg_confidence",
                          "accuracy", "ece", "n_total"}
    assert table["n_total"] == 450
    assert int(np.sum(table["counts"])) == 450
    assert table["avg_confidence"] == pytest.approx(table["accuracy"], abs=TOL)
    assert table["ece"] == pytest.approx(0.0, abs=TOL)
    assert table["adaptive"] is False
    # sparse binning: 20 bins over data living in two spikes leaves empty bins
    sparse = reliability_table(*_sparse_bin_data(), n_bins=20)
    assert int(np.sum(sparse["counts"])) == 8
    assert np.any(sparse["counts"] == 0)
    assert np.all(sparse["accuracy"][sparse["counts"] == 0] == 0.0)


# --------------------------------------------------------------------------- #
# multiclass metrics
# --------------------------------------------------------------------------- #
def test_multiclass_metrics_perfect_and_uniform() -> None:
    scores = np.array([[30.0, 0.0, 0.0], [0.0, 30.0, 0.0]])
    target = np.array([0, 1])
    assert multiclass_nll(scores, target) == pytest.approx(0.0, abs=1e-6)
    assert multiclass_brier(scores, target) == pytest.approx(0.0, abs=1e-6)
    k = 6
    uniform = np.zeros((4, k))
    tgt = np.arange(4) % k
    assert multiclass_nll(uniform, tgt) == pytest.approx(np.log(k), abs=TOL)
    expected_brier = ((1 - 1 / k) ** 2 + (k - 1) * (1 / k) ** 2) / 1.0
    assert multiclass_brier(uniform, tgt) == pytest.approx(expected_brier, abs=TOL)


def test_multiclass_temperature_and_absent_target() -> None:
    scores = np.array([[10.0, 0.0, -10.0]])
    target = np.array([2])  # confidently wrong
    cold = multiclass_nll(scores, target)
    hot = multiclass_nll(scores, target, temperature=1e9)
    assert np.isfinite(cold)
    assert hot == pytest.approx(np.log(3.0), abs=1e-6)  # T -> inf gives the uniform NLL
    assert hot < cold
    with pytest.raises(ValueError, match="target absent"):
        multiclass_nll(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, -1]))
    assert multiclass_nll(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0, -1]), strict=False) > 0
    p = 1.0 / (1.0 + np.exp(-1.0))  # softmax([1, 2])[1]
    assert multiclass_brier(
        np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1, -1]), mask=np.array([True, False])
    ) == pytest.approx(2.0 * (1.0 - p) ** 2, abs=TOL)


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def test_input_validation() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        top_label_ece(np.array([1.4, 0.2]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="binary"):
        top_label_ece(np.array([0.4, 0.2]), np.array([1.0, 0.5]))
    with pytest.raises(ValueError, match="differ in length"):
        brier_binary(np.array([0.4, 0.2, 0.9]), np.array([1, 0]))
    with pytest.raises(ValueError, match="n_bins"):
        top_label_ece(np.array([0.4]), np.array([1]), n_bins=0)
    with pytest.raises(ValueError, match="empty"):
        confidence_accuracy_gap(np.array([]), np.array([]))


def test_boolean_correctness_is_accepted() -> None:
    conf = np.array([0.9, 0.9, 0.2, 0.2])
    corr = np.array([True, True, False, True])
    assert brier_binary(conf, corr) == pytest.approx((0.01 + 0.01 + 0.04 + 0.64) / 4, abs=TOL)
    assert confidence_accuracy_gap(conf, corr) == pytest.approx(0.55 - 0.75, abs=TOL)
