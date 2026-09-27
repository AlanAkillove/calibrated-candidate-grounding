"""Temperature scaling (B2/C1, C2) and abstention thresholds (N0/N1)."""

from __future__ import annotations

import numpy as np
import pytest

from ccg.calibration.temperature import (
    DEFAULT_TEMP_BOUNDS,
    KAwareTemperature,
    TemperatureScaler,
    apply_temperature,
    apply_temperature_sets,
    fit_temperature,
    fit_temperature_k_aware,
    mean_nll,
    nll_at_temperature,
)
from ccg.calibration.thresholds import (
    ThresholdSelection,
    accept_mask,
    coverage_risk_curve,
    fit_margin_threshold,
    fit_max_conf_threshold,
    max_confidences,
    top1_top2_margins,
)
from ccg.models.cosine import CosineScorer


def synthetic_sets(num=400, k=10, error_rate=0.4, gap=6.8, seed=0):
    """Confident-and-wrong candidate sets - the overconfidence B2/C1 must fix.

    One candidate sits ``gap`` logits above the rest, so the softmax puts
    ``~0.99`` on it, while the expression is actually mis-ranked with probability
    ``error_rate``.  Flattening the logits (``T > 1``) then strictly lowers the
    candidate-set NLL, which is exactly what ``fit_temperature`` has to find.
    """
    rng = np.random.default_rng(seed)
    logits, targets = [], []
    for _ in range(num):
        row = np.zeros(k, dtype=np.float32)
        target = int(rng.integers(k))
        winner = int(rng.integers(k)) if float(rng.random()) < error_rate else target
        row[winner] = float(gap)
        logits.append(row)
        targets.append(target)
    return logits, np.asarray(targets, dtype=np.int64)


# ---------------------------------------------------------------------------
# NLL and the global temperature
# ---------------------------------------------------------------------------
def test_nll_at_temperature_hand_example():
    logits = [np.array([2.0, 0.0])]
    expected = -np.log(np.exp(2.0) / (np.exp(2.0) + 1.0))
    assert nll_at_temperature(logits, [0], 1.0) == pytest.approx(expected, rel=1e-6)
    assert mean_nll(logits, [0]) == pytest.approx(expected, rel=1e-6)
    # averaging over sets, and a wrong target gives the complementary probability
    both = [np.array([2.0, 0.0]), np.array([2.0, 0.0])]
    assert mean_nll(both, [0, 1]) == pytest.approx(
        0.5 * (expected + -np.log(1.0 / (np.exp(2.0) + 1.0))), rel=1e-6
    )
    # T -> inf is the uniform limit: NLL = log K
    assert mean_nll([np.array([2.0, 0.0, -3.0])], [0], temperature=1e6) == pytest.approx(
        np.log(3.0), abs=1e-3
    )


def test_fit_temperature_reduces_nll_on_overconfident_logits():
    logits, targets = synthetic_sets(error_rate=0.4)
    assert max_confidences(logits).mean() > 0.9, "the scorer is loudly confident"
    assert mean_nll(logits, targets, 1.0) > mean_nll(logits, targets, 4.0), "and wrong"
    before = mean_nll(logits, targets, 1.0)
    temperature = fit_temperature(logits, targets)
    after = mean_nll(logits, targets, temperature)
    assert temperature > 1.0, "overconfident margins must be softened by T > 1"
    assert after < before, "the whole point of B2: the fitted T lowers candidate-set NLL"
    # golden-section search finds (essentially) the grid optimum
    grid = np.exp(np.linspace(np.log(0.05), np.log(100.0), 400))
    best = min(mean_nll(logits, targets, t) for t in grid)
    assert after <= best + 1e-4
    # deterministic and reproducible from the calibration split alone
    assert fit_temperature(logits, targets) == pytest.approx(temperature, rel=1e-6)


def test_fit_temperature_may_sharpen_underconfident_logits():
    # a small margin plus label noise: T=1 is underconfident, so T < 1 helps
    rng = np.random.default_rng(7)
    logits, targets = [], []
    for _ in range(400):
        row = rng.normal(0.0, 0.05, size=3)
        target = int(rng.integers(3))
        row[target] += 0.2
        logits.append(row)
        targets.append(target)
    temperature = fit_temperature(logits, targets)
    assert temperature < 1.0, "an underconfident softmax must be sharpened"
    assert mean_nll(logits, targets, temperature) <= mean_nll(logits, targets, 1.0)


def test_temperature_input_validation():
    logits, targets = synthetic_sets(num=5, k=4)
    with pytest.raises(ValueError, match="K>=2"):
        fit_temperature([np.array([1.0])], [0])
    with pytest.raises(ValueError, match="target-absent"):
        fit_temperature(logits, [-1, *targets.tolist()[1:]])
    with pytest.raises(ValueError, match="but"):
        fit_temperature(logits, [0, 1])
    with pytest.raises(ValueError, match="no logit sets"):
        fit_temperature([], [])
    with pytest.raises(ValueError, match="non-finite"):
        fit_temperature([np.array([1.0, np.inf])], [0])
    with pytest.raises(ValueError, match="invalid temperature bounds"):
        fit_temperature(logits, targets, bounds=(0.0, 10.0))
    # a 1-D array is a single set
    assert 0.0 < fit_temperature(np.array([3.0, 0.0]), [0]) < 100.0


def test_apply_temperature_is_the_division_the_scorer_does():
    values = np.array([2.0, -1.0, 0.5], dtype=np.float32)
    scaled = apply_temperature(values, 4.0)
    np.testing.assert_allclose(scaled, values / 4.0, atol=1e-7)
    assert scaled.dtype == np.float32 and scaled.shape == (3,)
    with pytest.raises(ValueError, match="positive"):
        apply_temperature(values, 0.0)
    sets = apply_temperature_sets([values, values], 2.0)
    assert len(sets) == 2 and sets[0].shape == (3,)

    # B2 == CosineScorer with the fitted temperature attached (section 8)
    query = np.array([1.0, 0.5, -2.0], dtype=np.float32)
    candidates = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]], dtype=np.float32)
    raw = CosineScorer().score(query, candidates)
    np.testing.assert_allclose(
        CosineScorer(temperature=2.5).score(query, candidates), apply_temperature(raw, 2.5),
        atol=1e-6,
    )


def test_temperature_scaler_records_the_fit():
    logits, targets = synthetic_sets(num=200, k=6, seed=11)
    scaler = TemperatureScaler().fit(logits, targets)
    payload = scaler.to_dict()
    assert payload["fitted_on"] == "val_calib"
    assert payload["num_examples"] == 200
    assert payload["nll_after"] < payload["nll_before"]
    assert payload["temperature"] == pytest.approx(fit_temperature(logits, targets), rel=1e-4)
    assert payload["bounds"] == [DEFAULT_TEMP_BOUNDS[0], DEFAULT_TEMP_BOUNDS[1]]
    np.testing.assert_allclose(
        scaler.transform(logits[0]), apply_temperature(logits[0], scaler.temperature), atol=1e-6
    )
    probs = scaler.probabilities(logits[0])
    assert probs.sum() == pytest.approx(1.0) and probs.max() < 1.0
    # a temperature of 1.0 is a no-op fit for already-perfectly-scaled logits
    assert TemperatureScaler(temperature=2.0).temperature == 2.0


# ---------------------------------------------------------------------------
# C2: K-aware temperature, extrapolated (never fitted) at K=20/50
# ---------------------------------------------------------------------------
def test_k_aware_temperature_functional_form():
    model = KAwareTemperature(a=0.2, b=0.5)
    import math

    assert model.temperature_for(5) == pytest.approx(0.2 + 0.5 * math.log(5))
    assert model.temperature_for(50) > model.temperature_for(5), "b > 0 means T grows with K"
    with pytest.raises(ValueError, match="K must be >= 2"):
        model.temperature_for(1)
    tight = KAwareTemperature(a=0.2, b=5.0, bounds=(0.05, 1.0))
    assert tight.temperature_for(50) == 1.0, "clamping keeps T positive and finite"


def test_fit_temperature_k_aware_extrapolates_without_fitting():
    # K-dependent overconfidence: well-served at K=5, badly overconfident at K=10
    logits, targets = [], []
    for k, error_rate in ((5, 0.05), (10, 0.75)):
        rows, labels = synthetic_sets(num=200, k=k, error_rate=error_rate, seed=k)
        logits.extend(rows)
        targets.extend(labels.tolist())
    model = fit_temperature_k_aware(logits, targets)
    assert model.fitted_ks == (5, 10), "only the sizes seen in val_calib are fitted"
    assert model.num_examples == 400
    assert model.nll_after <= model.nll_before + 1e-9
    global_t = fit_temperature(logits, targets)
    assert model.nll_after <= mean_nll(logits, targets, global_t) + 1e-6
    assert model.b > 0.0 and model.temperature_for(10) > model.temperature_for(5)

    risk = model.extrapolation_risk(test_ks=(20, 50))
    assert risk["fitted_ks"] == [5, 10]
    assert risk["extrapolated_ks"] == [20, 50], "K=20/50 are extrapolated through the form"
    assert set(risk["temperatures"]) == {20, 50}
    payload = model.to_dict()
    assert payload["form"] == "T(K)=a+b*log(K)"
    assert payload["extrapolation"]["extrapolated_ks"] == [20, 50]

    per_set = model.predict(logits[:2], [5, 5])
    assert [item.size for item in per_set] == [5, 5]
    np.testing.assert_allclose(per_set[1], logits[1] / model.temperature_for(5), atol=1e-5)
    with pytest.raises(ValueError, match="aligned"):
        model.predict(logits[:3], [5, 10])
    probs = model.probabilities(logits[0])
    assert probs.sum() == pytest.approx(1.0)


def test_k_aware_falls_back_when_only_one_k_is_present():
    logits, targets = synthetic_sets(num=50, k=8, seed=3)
    model = fit_temperature_k_aware(logits, targets)
    assert model.b == 0.0
    assert model.a == pytest.approx(fit_temperature(logits, targets), rel=1e-4)
    assert any("not identifiable" in note for note in model.notes)
    assert model.temperature_for(8) == pytest.approx(model.temperature_for(50), rel=1e-2)
    # an explicit (wrong-length) ks list is rejected rather than broadcast
    with pytest.raises(ValueError, match="K values"):
        fit_temperature_k_aware(logits, targets, ks=[8] * 49)


# ---------------------------------------------------------------------------
# N0 / N1 abstention thresholds
# ---------------------------------------------------------------------------
def selective_data(num=1000):
    """Deterministic monotone toy: confidence ``i/n``, correct above 0.2.

    Monotone so that "max coverage under a risk budget" has a closed-form answer
    and the assertions below never depend on a random seed.
    """
    confidence = (np.arange(num) + 1.0) / num
    correct = (confidence >= 0.2).astype(np.int64)
    return confidence, correct


def test_max_confidences_and_margins_definitions():
    logits = [np.array([2.0, 0.0, 0.0]), np.array([0.0, 5.0, 1.0])]
    conf = max_confidences(logits)
    p = np.exp(np.array([2.0, 0.0, 0.0]) - 2.0)
    assert conf[0] == pytest.approx(float(p.max() / p.sum()))
    assert conf[1] > conf[0]
    margins = top1_top2_margins(logits)
    probs = np.exp(np.array([2.0, 0.0, 0.0]) - 2.0)
    probs = probs / probs.sum()
    ordered = np.sort(probs)[::-1]
    assert margins[0] == pytest.approx(float(ordered[0] - ordered[1]))
    raw = top1_top2_margins(logits, kind="score")
    assert raw[0] == pytest.approx(2.0) and raw[1] == pytest.approx(4.0)
    with pytest.raises(ValueError, match="kind must be"):
        top1_top2_margins(logits, kind="cosine")
    with pytest.raises(ValueError, match="at least two"):
        top1_top2_margins([np.array([1.0])])


def test_coverage_risk_curve_is_monotone_and_honest():
    scores = np.array([0.9, 0.8, 0.4, 0.2])
    labels = np.array([1, 0, 1, 0])
    curve = coverage_risk_curve(scores, labels, thresholds=[0.0, 0.3, 0.85, 1.0])
    np.testing.assert_allclose(curve["coverage"], [1.0, 0.75, 0.25, 0.0])
    assert np.isnan(curve["risk"][-1]), "an empty acceptance set has no risk"
    np.testing.assert_allclose(curve["risk"][:3], [0.5, 1 / 3, 0.0])
    assert curve["num_examples"][0] == 4
    with pytest.raises(ValueError, match="must align"):
        coverage_risk_curve(scores, labels[:2])
    with pytest.raises(ValueError, match="at least one"):
        coverage_risk_curve([], [])
    with pytest.raises(ValueError, match="0/1"):
        coverage_risk_curve(scores, np.array([0.5, 0.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="non-finite"):
        coverage_risk_curve(np.array([np.nan]), np.array([1]))


def test_fit_max_conf_threshold_hits_the_target_coverage():
    confidence, correct = selective_data()
    for target in (0.5, 0.8, 0.95):
        selection = fit_max_conf_threshold(confidence, correct, target_coverage=target)
        assert abs(selection.coverage - target) <= 0.02, f"target {target} -> {selection.coverage}"
        assert selection.satisfied and selection.criterion == "target_coverage"
        assert selection.target_value == pytest.approx(target)
        assert selection.num_accepted == round(selection.coverage * selection.num_examples)
        accepted = accept_mask(confidence, selection.threshold)
        assert accepted.mean() == pytest.approx(selection.coverage)
        assert selection.accuracy == pytest.approx(1.0 - selection.risk)
        assert selection.score_kind == "max_confidence"
        assert selection.to_dict()["threshold"] == pytest.approx(selection.threshold)


def test_threshold_is_fitted_only_where_the_risk_budget_allows():
    confidence, correct = selective_data()
    selection = fit_max_conf_threshold(confidence, correct, target_risk=0.1)
    assert selection.satisfied and selection.risk <= 0.1 + 1e-12
    assert 0.8 < selection.coverage < 1.0, "abstention must actually reject something"
    # a stricter budget can only cost coverage
    strict = fit_max_conf_threshold(confidence, correct, target_risk=0.0)
    assert strict.satisfied and strict.accuracy == pytest.approx(1.0)
    assert strict.coverage == pytest.approx(0.8, abs=0.01)
    assert strict.coverage <= selection.coverage + 1e-9
    # an impossible budget is reported (satisfied=False), never hidden
    always_wrong = np.zeros_like(correct)
    hopeless = fit_max_conf_threshold(confidence, always_wrong, target_risk=0.5)
    assert hopeless.satisfied is False and hopeless.notes
    assert hopeless.risk == pytest.approx(1.0)


def test_margin_threshold_shares_the_selection_machinery():
    rng = np.random.default_rng(5)
    logits, targets = [], []
    for _ in range(400):  # continuous margins, so any coverage is reachable
        row = rng.normal(0.0, 1.0, size=10)
        target = int(rng.integers(10))
        row[target] += rng.uniform(0.0, 4.0)
        logits.append(row)
        targets.append(target)
    margins = top1_top2_margins(logits)
    correct = np.array([int(np.argmax(row)) == target for row, target in zip(logits, targets)])
    selection = fit_margin_threshold(margins, correct, target_coverage=0.7)
    assert isinstance(selection, ThresholdSelection)
    assert abs(selection.coverage - 0.7) <= 0.02 and selection.satisfied
    assert selection.score_kind == "margin"
    assert accept_mask(margins, selection.threshold).mean() == pytest.approx(
        selection.coverage, abs=1e-9
    )
    # accepting on a large margin really is more accurate than accepting everything
    assert selection.accuracy > float(np.mean(correct))


def test_threshold_argument_exclusivity():
    confidence, correct = selective_data(num=100)
    with pytest.raises(ValueError, match="exactly one"):
        fit_max_conf_threshold(confidence, correct)
    with pytest.raises(ValueError, match="exactly one"):
        fit_max_conf_threshold(confidence, correct, target_risk=0.1, target_coverage=0.5)
    with pytest.raises(ValueError, match="target_coverage must be"):
        fit_max_conf_threshold(confidence, correct, target_coverage=0.0)
    with pytest.raises(ValueError, match="target_risk must be"):
        fit_max_conf_threshold(confidence, correct, target_risk=1.0)
    with pytest.raises(ValueError, match="min_coverage"):
        fit_max_conf_threshold(confidence, correct, target_coverage=0.5, min_coverage=0.0)
