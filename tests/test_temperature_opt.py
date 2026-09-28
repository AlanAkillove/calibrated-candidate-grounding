"""Tests for :mod:`ccg.calibration.temperature_opt` (Phase 0A.1).

The synthetic datasets are built so the *true* optimal temperature is known
exactly: candidate sets share a fixed score vector ``s`` and the target labels
are replicated so that their empirical distribution equals
``softmax(s / T_true)``.  The NLL objective is then a cross-entropy whose
unique minimiser is ``T_true`` (up to the small error from rounding the
replication counts), which lets the fitter's recovery, interior detection,
one-shot expansion and warning be checked without any guessing.
"""

from __future__ import annotations

import numpy as np
import pytest

from ccg.calibration.temperature_opt import (
    TEMPERATURE_OPTIMIZATION_WARNING,
    TemperatureFit,
    fit_temperature_log_space,
    nll_of_temperature,
)

#: 8 distinct score values (so the softmax is identifiable from the counts).
_BASE = np.linspace(-1.0, 1.0, 8)


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(scores, dtype=np.float64) / float(temperature)
    scaled = scaled - scaled.max()
    exp = np.exp(scaled)
    return exp / exp.sum()


def _counted_sets(t_true: float, n_total: int = 20000, scale: float = 2.0):
    """``(sets, labels)`` whose empirical target distribution is softmax(s/T_true).

    ``s `` is scaled with ``T_true`` so that ``s / T_true = scale * _BASE`` is
    ``O(1)`` for every ``T_true`` - this keeps the softmax well spread (never a
    degenerate one-hot) independently of the temperature magnitude.
    """
    scores = (_BASE * (scale * float(t_true))).astype(np.float64)
    q = _softmax(scores, t_true)
    counts = np.maximum(np.round(q * int(n_total)).astype(np.int64), 1)
    labels = np.repeat(np.arange(scores.size), counts)
    sets = [scores] * int(labels.size)  # shared reference keeps memory small
    return sets, labels


def test_fit_recovers_known_interior_temperature():
    sets, labels = _counted_sets(0.5)
    fit = fit_temperature_log_space(sets, labels)

    assert isinstance(fit, TemperatureFit)
    assert fit.interior is True
    assert fit.warning is None
    assert fit.bounds_expanded is False
    assert fit.n_expansions == 0
    assert fit.decades_to_low >= 1.0 and fit.decades_to_high >= 1.0
    # recovered within 10% (the requirement)
    assert abs(fit.temperature - 0.5) / 0.5 < 0.10
    # temperature scaling must never increase the objective
    assert fit.nll_after <= fit.nll_before + 1e-9
    assert fit.n_sets == labels.size


def test_boundary_optimum_triggers_one_expansion():
    # true optimum 1e-4 is below the initial t_min = 1e-3 -> one expansion
    sets, labels = _counted_sets(1e-4)
    fit = fit_temperature_log_space(sets, labels)

    assert fit.bounds_expanded is True
    assert fit.n_expansions == 1
    assert fit.bounds_initial == (pytest.approx(1e-3), pytest.approx(10.0))
    assert fit.bounds[0] == pytest.approx(1e-5)
    assert fit.bounds[1] == pytest.approx(1000.0)
    # the optimum sits near the true 1e-4
    assert 5e-5 <= fit.temperature <= 2e-4


def test_beyond_expanded_bounds_emits_warning():
    # true optimum 1e-8 is below even the expanded t_min = 1e-5 -> warning
    sets, labels = _counted_sets(1e-8)
    fit = fit_temperature_log_space(sets, labels)

    assert fit.bounds_expanded is True
    assert fit.n_expansions == 1
    assert fit.interior is False
    assert fit.warning == TEMPERATURE_OPTIMIZATION_WARNING
    assert fit.warning_detail is not None
    assert "LOWER" in fit.warning_detail
    # pinned at the expanded lower bound
    assert fit.temperature == pytest.approx(1e-5, rel=0.5)


def test_expand_once_disabled_keeps_initial_bounds():
    sets, labels = _counted_sets(1e-4)
    fit = fit_temperature_log_space(sets, labels, expand_once=False)
    assert fit.bounds_expanded is False
    assert fit.n_expansions == 0
    assert fit.bounds == (pytest.approx(1e-3), pytest.approx(10.0))
    assert fit.interior is False
    assert fit.warning == TEMPERATURE_OPTIMIZATION_WARNING


def test_to_dict_roundtrips_fields():
    sets, labels = _counted_sets(0.5)
    payload = fit_temperature_log_space(sets, labels).to_dict()
    for key in (
        "temperature",
        "u_opt",
        "bounds",
        "bounds_initial",
        "bounds_expanded",
        "n_expansions",
        "interior",
        "decades_to_low",
        "decades_to_high",
        "warning",
        "objective",
        "n_sets",
        "nll_before",
        "nll_after",
        "method",
    ):
        assert key in payload
    assert payload["method"] == "scipy.optimize.minimize_scalar(bounded) on u=lnT"


def test_nll_of_temperature_flat_limit():
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    value = nll_of_temperature([scores], [0], temperature=1e6)
    assert value == pytest.approx(np.log(4.0), rel=1e-3)


def test_nll_of_temperature_sharp_limit():
    scores = np.array([5.0, 0.0, -5.0])
    # T -> 0 drives the target probability to 1 (when the target is the argmax)
    value = nll_of_temperature([scores], [0], temperature=1e-3)
    assert value == pytest.approx(0.0, abs=1e-6)
    # target == a non-argmax candidate -> large NLL
    other = nll_of_temperature([scores], [2], temperature=1e-3)
    assert other > 10.0


def test_nll_of_temperature_supports_mixed_set_sizes():
    value = nll_of_temperature(
        [np.array([1.0, 0.0]), np.array([0.5, 0.2, 0.1])], [1, 2], 1.0
    )
    assert value > 0.0


def test_nll_of_temperature_empty_raises():
    with pytest.raises(ValueError):
        nll_of_temperature([], [], 1.0)


def test_nll_of_temperature_bad_temperature_raises():
    with pytest.raises(ValueError):
        nll_of_temperature([np.array([1.0, 2.0])], [0], 0.0)


def test_nll_of_temperature_target_out_of_range_raises():
    with pytest.raises(ValueError):
        nll_of_temperature([np.array([1.0, 2.0])], [5], 1.0)


def test_fit_invalid_bounds_raise():
    sets, labels = _counted_sets(0.5)
    with pytest.raises(ValueError):
        fit_temperature_log_space(sets, labels, t_min=10.0, t_max=1.0)
    with pytest.raises(ValueError):
        fit_temperature_log_space(sets, labels, expansion=0.5)
