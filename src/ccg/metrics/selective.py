"""Selective-prediction (risk-coverage) metrics.

Contract functions: :func:`risk_coverage_curve`, :func:`aurc`,
:func:`selective_accuracy`, :func:`risk_at_coverage`.

Definition
----------
The abstention policy accepts the ``n`` samples with the highest confidence, for
every ``n = 1 .. N``. Coverage is ``phi = n / N`` and

    risk(phi) = 1 - accuracy(accepted samples)

so **risk = 1 - accuracy** everywhere in this project (lower is better). The
returned curve therefore has one point per achievable coverage level and is not
smoothed; ties in confidence are resolved by input order (stable sort), which
keeps the curve deterministic.

Degenerate behaviours that are *expected*, not bugs:

* all samples correct -> ``risk`` is all zeros and ``AURC == 0``;
* all samples wrong -> ``risk`` is all ones and ``AURC == 1``;
* constant confidence -> the ordering degenerates to the input order, and when
  correctness is constant as well the curve is flat at ``1 - accuracy``: the
  selective accuracy equals the full-set accuracy at every coverage level.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ccg.metrics._validation import (
    as_correctness,
    require_same_length,
    to_numpy_1d,
    to_probability_1d,
    trapz,
)

__all__ = [
    "aurc",
    "e_aurc",
    "oracle_aurc",
    "rer_at_coverage",
    "risk_at_coverage",
    "risk_coverage_curve",
    "selective_accuracy",
    "selective_risk",
]


def _validate_pairs(confidence: Any, correctness: Any) -> tuple[NDArray, NDArray]:
    """Return ``(confidence, correctness)`` as aligned float arrays in ``[0, 1]``."""
    conf = to_probability_1d(confidence, "confidence")
    corr = as_correctness(correctness, "correctness")
    require_same_length(conf, corr, "confidence", "correctness")
    return conf, corr


def risk_coverage_curve(
    confidence: Any,
    correctness: Any,
) -> tuple[NDArray, NDArray]:
    """Return ``(coverage, risk)`` of the confidence-ordered selective predictor.

    Both arrays have length ``N`` (number of samples); ``coverage[i] = (i+1)/N``
    and ``risk[i] = 1 - mean(correctness of the accepted top-(i+1) samples)``.
    """
    conf, corr = _validate_pairs(confidence, correctness)
    order = np.argsort(-conf, kind="stable")
    corr_sorted = corr[order]
    n = np.arange(1, corr_sorted.shape[0] + 1, dtype=np.float64)
    running_correct = np.cumsum(corr_sorted)
    coverage = n / float(corr_sorted.shape[0])
    risk = 1.0 - running_correct / n
    return coverage.astype(np.float64), risk.astype(np.float64)


def aurc(coverage: Any, risk: Any) -> float:
    """Area under the risk-coverage curve via trapezoidal integration.

    ``coverage`` must be monotonically ordered (ascending or descending - the
    absolute value of the integral is returned so both work). No origin point is
    injected: the area is taken over exactly the supplied points, which is the
    convention used for every comparison in this project.
    """
    cov = to_numpy_1d(coverage, "coverage")
    rsk = to_numpy_1d(risk, "risk")
    if cov.shape[0] != rsk.shape[0]:
        raise ValueError(
            f"`coverage` ({cov.shape[0]}) and `risk` ({rsk.shape[0]}) differ in length"
        )
    if cov.shape[0] == 1:
        return float(rsk[0])
    if not np.all(np.isfinite(rsk)):
        raise ValueError("`risk` contains NaN or inf values")
    if not np.all(np.diff(cov) >= 0) and not np.all(np.diff(cov) <= 0):
        raise ValueError("`coverage` must be monotonically ordered")
    order = np.argsort(cov, kind="stable")
    return abs(trapz(rsk[order], cov[order]))


def selective_accuracy(confidence: Any, correctness: Any, coverage_level: float) -> float:
    """Accuracy restricted to the most confident ``coverage_level`` fraction.

    ``coverage_level`` in ``(0, 1]``. The number of accepted samples is
    ``max(1, ceil(coverage_level * N))`` so that the realised coverage is never
    below the requested one; ``coverage_level = 1.0`` reproduces plain accuracy.
    """
    conf, corr = _validate_pairs(confidence, correctness)
    level = float(coverage_level)
    if not np.isfinite(level) or level <= 0.0 or level > 1.0:
        raise ValueError(f"`coverage_level` must be in (0, 1], got {coverage_level}")
    n_total = corr.shape[0]
    n_keep = max(1, int(np.ceil(level * n_total)))
    order = np.argsort(-conf, kind="stable")[:n_keep]
    return float(np.mean(corr[order]))


def selective_risk(confidence: Any, correctness: Any, coverage_level: float) -> float:
    """``1 - selective_accuracy`` at the requested coverage level."""
    return 1.0 - selective_accuracy(confidence, correctness, coverage_level)


def risk_at_coverage(confidence: Any, correctness: Any, coverage_level: float) -> float:
    """Risk (= 1 - accuracy) at a fixed coverage level, e.g. 0.5/0.8/0.9/0.95.

    This is the ``risk@50%`` / ``risk@80%`` / ``risk@90%`` / ``risk@95%`` family
    of headline numbers used in the Phase-0 report.
    """
    return selective_risk(confidence, correctness, coverage_level)


# ---------------------------------------------------------------------------
# base-rate-aware / accuracy-normalised selective metrics (Phase 0A.1)
# ---------------------------------------------------------------------------
def oracle_aurc(error_rate: float) -> float:
    """AURC of an *oracle* confidence ranking at a given full-set error rate.

    When samples are ranked by their true correctness (all correct first), the
    risk-coverage curve is the best achievable one and its area has a closed
    form in the full-set error rate ``r``:

        oracle_aurc(r) = r + (1 - r) * ln(1 - r)

    with the conventions ``oracle_aurc(0) = 0`` and ``oracle_aurc(1) = 1`` (the
    latter is the limiting value; ``0 * ln(0)`` is taken to be ``0``).  This is
    the accuracy-normalisation baseline that removes the AURC / base-error-rate
    coupling: a *worse* error rate depresses the curve, so raw AURC alone cannot
    be compared across K.  ``ln(1 - r)`` is evaluated with :func:`numpy.log1p`
    for accuracy near ``r = 0``.

    Parameters
    ----------
    error_rate:
        Full-set error rate ``r = 1 - accuracy`` in ``[0, 1]``.
    """
    r = float(error_rate)
    if not np.isfinite(r) or r < 0.0 or r > 1.0:
        raise ValueError(f"`error_rate` must be in [0, 1], got {error_rate}")
    if r == 0.0:
        return 0.0
    if r == 1.0:
        return 1.0
    return float(r + (1.0 - r) * np.log1p(-r))


def e_aurc(aurc_value: float, accuracy: float) -> float:
    """Excess AURC over the oracle bound: ``AURC - oracle_aurc(1 - accuracy)``.

    ``E-AURC >= 0``: how far the confidence ordering is from the best possible
    ordering *at that accuracy*.  Subtracting the oracle term removes the
    monotone dependence of raw AURC on the base error rate, so two candidate-set
    sizes (or two variants) with different accuracies become comparable.
    """
    value = float(aurc_value)
    if not np.isfinite(value):
        raise ValueError(f"`aurc_value` must be finite, got {aurc_value}")
    return float(value - oracle_aurc(1.0 - float(accuracy)))


def rer_at_coverage(confidence: Any, correctness: Any, coverage_level: float) -> float:
    """Risk-Erosion Ratio ``(R1 - Rc) / R1`` at a coverage level.

    ``R1 = 1 - mean(correctness)`` is the risk of the full set (no abstention)
    and ``Rc = risk_at_coverage(confidence, correctness, coverage_level)`` is the
    risk of the most-confident ``coverage_level`` fraction.  The ratio is the
    *relative* risk removed by abstention: ``1.0`` when the accepted subset is
    error-free, ``0.0`` when abstention removes no risk.

    If ``R1 == 0`` (every sample is correct) there is no base error to erode and
    no measurable benefit, so ``0.0`` is returned by convention (the ratio is
    otherwise ``0 / 0``).
    """
    conf, corr = _validate_pairs(confidence, correctness)
    base_risk = 1.0 - float(np.mean(corr))
    if base_risk == 0.0:
        return 0.0
    risk_at_c = risk_at_coverage(conf, corr, coverage_level)
    return float((base_risk - risk_at_c) / base_risk)
